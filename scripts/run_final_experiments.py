import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_csv_table, report_image
from src.data.segmentation_dataset import MVTecSegmentationDataset, subset_rows
from src.training_utils import compute_pos_weight, run_epoch, SegmentationLoss
from src.visualization import (
    ensure_dir,
    save_comparison_plot,
    save_loss_components_curve,
    save_prediction_figure,
    save_training_curve,
    save_validation_error_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run final U-Net experiments for multiple categories")
    parser.add_argument("--categories", default="bottle,capsule,metal_nut,pill")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs_per_category", type=int, default=12)
    parser.add_argument("--epochs_all_in_one", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--example_count", type=int, default=12)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=0.6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_root", default="results/final")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices(length: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(length * val_ratio))
    return indices[val_size:], indices[:val_size]


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_category_splits(args) -> dict[str, dict]:
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    category_data: dict[str, dict] = {}
    for offset, category in enumerate(categories):
        dataset = MVTecSegmentationDataset(args.data_root, category, image_size=args.image_size)
        train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed + offset)
        category_data[category] = {
            "dataset": dataset,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }
    return category_data


def train_model(
    task_name: str,
    tags: list[str],
    model_path: Path,
    output_dir: Path,
    train_dataset,
    train_rows: list[dict],
    dataset_rows: list[dict],
    val_dataset,
    val_rows: list[dict],
    pos_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    bce_weight: float,
    grad_clip: float,
    device: torch.device,
    params: dict,
) -> tuple[UNet, list[dict], int, float]:
    task = init_task("Conveyor Defect Detection", task_name, tags, params=params)
    ensure_dir(output_dir)
    ensure_dir(model_path.parent)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = UNet().to(device)
    criterion = SegmentationLoss(pos_weight=pos_weight, bce_weight=bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    history_rows: list[dict] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_bce_losses: list[float] = []
    val_bce_losses: list[float] = []
    train_dice_losses: list[float] = []
    val_dice_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        for _ in tqdm(range(1), desc=f"{task_name} | epoch {epoch + 1}/{epochs}", leave=False):
            train_stats = run_epoch(
                model, train_loader, optimizer, criterion, device, train=True, grad_clip=grad_clip
            )
            val_stats = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        scheduler.step(val_stats.total_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_stats.total_loss)
        val_losses.append(val_stats.total_loss)
        train_bce_losses.append(train_stats.bce_loss)
        val_bce_losses.append(val_stats.bce_loss)
        train_dice_losses.append(train_stats.dice_loss)
        val_dice_losses.append(val_stats.dice_loss)

        improved = val_stats.total_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_stats.total_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_stats.total_loss, 6),
            "train_error": round(train_stats.total_loss, 6),
            "train_bce": round(train_stats.bce_loss, 6),
            "train_dice_loss": round(train_stats.dice_loss, 6),
            "val_loss": round(val_stats.total_loss, 6),
            "validation_error": round(val_stats.total_loss, 6),
            "val_bce": round(val_stats.bce_loss, 6),
            "val_dice_loss": round(val_stats.dice_loss, 6),
            "learning_rate": f"{current_lr:.8f}",
            "best_model": "yes" if improved else "no",
        }
        history_rows.append(row)
        task.get_logger().report_scalar("Loss", "train", train_stats.total_loss, epoch)
        task.get_logger().report_scalar("Loss", "validation", val_stats.total_loss, epoch)
        task.get_logger().report_scalar("Validation Error", task_name, val_stats.total_loss, epoch)
        task.get_logger().report_scalar("BCE", "train", train_stats.bce_loss, epoch)
        task.get_logger().report_scalar("BCE", "validation", val_stats.bce_loss, epoch)
        task.get_logger().report_scalar("Dice loss", "train", train_stats.dice_loss, epoch)
        task.get_logger().report_scalar("Dice loss", "validation", val_stats.dice_loss, epoch)
        task.get_logger().report_scalar("Learning rate", "optimizer", current_lr, epoch)
        print(
            f"{task_name}: epoch {epoch + 1}/{epochs}, "
            f"train_loss={train_stats.total_loss:.4f}, "
            f"val_loss={val_stats.total_loss:.4f}"
        )

        if stale_epochs >= patience:
            print(
                f"{task_name}: early stopping at epoch {epoch + 1} "
                f"because validation loss stopped improving."
            )
            break

    history_path = output_dir / "training_history.csv"
    curve_path = output_dir / "training_curve.png"
    validation_curve_path = output_dir / "validation_error_curve.png"
    components_curve_path = output_dir / "loss_components_curve.png"
    dataset_summary_path = output_dir / "dataset_summary.csv"
    train_split_path = output_dir / "train_split.csv"
    val_split_path = output_dir / "val_split.csv"

    save_csv(history_rows, history_path)
    save_csv(dataset_rows, dataset_summary_path)
    save_csv(train_rows, train_split_path)
    save_csv(val_rows, val_split_path)
    save_training_curve(train_losses, val_losses, curve_path)
    save_validation_error_curve(val_losses, validation_curve_path, best_epoch=best_epoch)
    save_loss_components_curve(
        train_bce_losses,
        val_bce_losses,
        train_dice_losses,
        val_dice_losses,
        components_curve_path,
    )

    report_csv_table(task, "Training History", "loss", history_path)
    report_csv_table(task, "Dataset Summary", "dataset", dataset_summary_path)
    report_csv_table(task, "Train Split", "dataset", train_split_path)
    report_csv_table(task, "Validation Split", "dataset", val_split_path)
    report_image(task, "Training Curve", "plots", curve_path)
    report_image(task, "Validation Error Curve", "plots", validation_curve_path)
    report_image(task, "Loss Components", "plots", components_curve_path)

    task.close()
    return model, history_rows, best_epoch, best_val_loss


def dice_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def evaluate_model(
    task_name: str,
    tags: list[str],
    model_path: Path,
    output_dir: Path,
    dataset: MVTecSegmentationDataset,
    val_indices: list[int],
    threshold: float,
    batch_size: int,
    device: torch.device,
    example_count: int,
    params: dict,
) -> dict:
    task = init_task("Conveyor Defect Detection", task_name, tags, params=params)
    ensure_dir(output_dir)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    prediction_rows: list[dict] = []
    dice_total = 0.0
    iou_total = 0.0
    examples_dir = output_dir / "examples"
    ensure_dir(examples_dir)
    saved_examples = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

            dice_total += dice_score(predictions, masks).item()
            iou_total += iou_score(predictions, masks).item()

            for idx in range(images.size(0)):
                image_name = Path(batch["image_path"][idx]).name
                defect_type = batch["defect_type"][idx]
                predicted_ratio = float(predictions[idx].mean().item())
                target_ratio = float(masks[idx].mean().item())
                prediction_rows.append(
                    {
                        "image_name": image_name,
                        "defect_type": defect_type,
                        "predicted_defect_ratio": round(predicted_ratio, 6),
                        "target_defect_ratio": round(target_ratio, 6),
                    }
                )

                if saved_examples < example_count:
                    example_path = examples_dir / f"example_{saved_examples + 1:02d}.png"
                    save_prediction_figure(
                        image=images[idx].cpu(),
                        target_mask=masks[idx].cpu(),
                        predicted_mask=predictions[idx].cpu(),
                        path=example_path,
                        title=f"{task_name}: {image_name}",
                    )
                    report_image(task, "Segmentation Examples", f"example_{saved_examples + 1}", example_path)
                    saved_examples += 1

    metrics = {
        "dice": round(dice_total / len(val_loader), 6),
        "iou": round(iou_total / len(val_loader), 6),
    }
    summary_rows = [
        {
            "metric": "Dice",
            "value": metrics["dice"],
            "explanation": "Overlap between predicted mask and ground-truth mask. Closer to 1 is better.",
        },
        {
            "metric": "IoU",
            "value": metrics["iou"],
            "explanation": "Intersection over Union for segmentation masks. Closer to 1 is better.",
        },
        {
            "metric": "Threshold",
            "value": threshold,
            "explanation": "Probability cutoff used to convert the predicted map into a binary mask.",
        },
    ]

    summary_path = output_dir / "evaluation_summary.csv"
    prediction_path = output_dir / "prediction_table.csv"
    val_split_path = output_dir / "val_split.csv"
    save_csv(summary_rows, summary_path)
    save_csv(prediction_rows, prediction_path)
    save_csv(subset_rows(dataset, val_indices), val_split_path)

    report_csv_table(task, "Evaluation Summary", "metrics", summary_path)
    report_csv_table(task, "Prediction Table", "predictions", prediction_path)
    report_csv_table(task, "Validation Split", "dataset", val_split_path)
    task.get_logger().report_scalar("Segmentation Metrics", "Dice", metrics["dice"], 0)
    task.get_logger().report_scalar("Segmentation Metrics", "IoU", metrics["iou"], 0)

    task.close()
    return metrics


def main() -> None:
    args = parse_args()
    device = get_device()
    root_dir = Path(args.output_root)
    ensure_dir(root_dir)

    category_data = load_category_splits(args)
    comparison_rows: list[dict] = []
    split_manifest: dict[str, dict] = {}

    for category, info in category_data.items():
        split_manifest[category] = {
            "train_indices": info["train_indices"],
            "val_indices": info["val_indices"],
        }

    for category, info in category_data.items():
        per_train_dir = root_dir / "per_category" / category / "train"
        per_eval_dir = root_dir / "per_category" / category / "eval"
        model_path = Path("models") / f"unet_per_category_{category}.pth"
        train_subset = Subset(info["dataset"], info["train_indices"])
        train_rows = subset_rows(info["dataset"], info["train_indices"])

        pos_weight = compute_pos_weight(info["dataset"], info["train_indices"], device)
        train_model(
            task_name=f"Final Train U-Net ({category}, per-category)",
            tags=["final", "unet", "segmentation", "per-category", category],
            model_path=model_path,
            output_dir=per_train_dir,
            train_dataset=train_subset,
            train_rows=train_rows,
            dataset_rows=info["dataset"].summary_rows(),
            val_dataset=Subset(info["dataset"], info["val_indices"]),
            val_rows=subset_rows(info["dataset"], info["val_indices"]),
            pos_weight=pos_weight,
            epochs=args.epochs_per_category,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            bce_weight=args.bce_weight,
            grad_clip=args.grad_clip,
            device=device,
            params={**vars(args), "category": category, "model_type": "per-category"},
        )

        metrics = evaluate_model(
            task_name=f"Final Evaluate U-Net ({category}, per-category)",
            tags=["final", "unet", "segmentation", "evaluation", "per-category", category],
            model_path=model_path,
            output_dir=per_eval_dir,
            dataset=info["dataset"],
            val_indices=info["val_indices"],
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            example_count=args.example_count,
            params={**vars(args), "category": category, "model_type": "per-category"},
        )

        comparison_rows.append(
            {
                "category": category,
                "model_type": "per-category",
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "train_images": len(info["train_indices"]),
                "val_images": len(info["val_indices"]),
            }
        )

    combined_train_subsets = []
    combined_val_subsets = []
    combined_train_rows: list[dict] = []
    combined_val_rows: list[dict] = []
    combined_dataset_rows: list[dict] = []
    combined_positive_weight_indices: list[tuple[MVTecSegmentationDataset, int]] = []
    for category, info in category_data.items():
        combined_train_subsets.append(Subset(info["dataset"], info["train_indices"]))
        combined_val_subsets.append(Subset(info["dataset"], info["val_indices"]))
        combined_train_rows.extend(
            [
                {"category": category, **row}
                for row in subset_rows(info["dataset"], info["train_indices"])
            ]
        )
        combined_val_rows.extend(
            [
                {"category": category, **row}
                for row in subset_rows(info["dataset"], info["val_indices"])
            ]
        )
        for row in info["dataset"].summary_rows():
            combined_dataset_rows.append({"category": category, **row})
        combined_positive_weight_indices.extend(
            (info["dataset"], index) for index in info["train_indices"]
        )

    total_positive = 0.0
    total_negative = 0.0
    for dataset, index in combined_positive_weight_indices:
        mask = dataset[index]["mask"]
        total_positive += float(mask.sum().item())
        total_negative += float(mask.numel() - mask.sum().item())
    all_pos_weight = torch.tensor(
        total_negative / max(total_positive, 1.0),
        dtype=torch.float32,
        device=device,
    )

    all_model_path = Path("models") / "unet_all_in_one.pth"
    train_model(
        task_name="Final Train U-Net (all-in-one)",
        tags=["final", "unet", "segmentation", "all-in-one"],
        model_path=all_model_path,
        output_dir=root_dir / "all_in_one" / "train",
        train_dataset=ConcatDataset(combined_train_subsets),
        train_rows=combined_train_rows,
        dataset_rows=combined_dataset_rows,
        val_dataset=ConcatDataset(combined_val_subsets),
        val_rows=combined_val_rows,
        pos_weight=all_pos_weight,
        epochs=args.epochs_all_in_one,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        bce_weight=args.bce_weight,
        grad_clip=args.grad_clip,
        device=device,
        params={**vars(args), "model_type": "all-in-one"},
    )

    for category, info in category_data.items():
        metrics = evaluate_model(
            task_name=f"Final Evaluate U-Net ({category}, all-in-one)",
            tags=["final", "unet", "segmentation", "evaluation", "all-in-one", category],
            model_path=all_model_path,
            output_dir=root_dir / "all_in_one" / category / "eval",
            dataset=info["dataset"],
            val_indices=info["val_indices"],
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            example_count=args.example_count,
            params={**vars(args), "category": category, "model_type": "all-in-one"},
        )
        comparison_rows.append(
            {
                "category": category,
                "model_type": "all-in-one",
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "train_images": sum(len(item["train_indices"]) for item in category_data.values()),
                "val_images": len(info["val_indices"]),
            }
        )

    comparison_dir = root_dir / "comparison"
    ensure_dir(comparison_dir)
    comparison_path = comparison_dir / "final_comparison.csv"
    comparison_plot_path = comparison_dir / "final_comparison.png"
    split_manifest_path = comparison_dir / "split_manifest.json"
    save_csv(comparison_rows, comparison_path)
    save_comparison_plot(comparison_rows, comparison_plot_path)
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)

    task = init_task(
        "Conveyor Defect Detection",
        "Final Comparison Table (U-Net)",
        ["final", "unet", "segmentation", "comparison"],
        params=vars(args),
    )
    report_csv_table(task, "Final Comparison", "summary", comparison_path)
    report_image(task, "Final Comparison Plot", "summary", comparison_plot_path)
    task.close()

    print("Final experiments completed.")
    print(f"Comparison table saved to {comparison_path}")


if __name__ == "__main__":
    main()
