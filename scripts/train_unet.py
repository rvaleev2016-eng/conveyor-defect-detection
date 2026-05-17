import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_csv_table, report_image, upload_if_exists
from src.data.segmentation_dataset import MVTecSegmentationDataset, subset_rows
from src.training_utils import compute_pos_weight, run_epoch, SegmentationLoss
from src.visualization import (
    ensure_dir,
    save_loss_components_curve,
    save_training_curve,
    save_validation_error_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for defect segmentation")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "rmsprop", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=0.6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_root", default="results")
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
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Train split is empty. Reduce val_ratio or use more data.")
    return train_indices, val_indices


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = get_device()
    task = init_task(
        project_name="Conveyor Defect Detection",
        task_name=f"Train U-Net ({args.category})",
        tags=["unet", "segmentation", "train", args.category],
        params=vars(args),
    )

    dataset = MVTecSegmentationDataset(
        root=args.data_root,
        category=args.category,
        image_size=args.image_size,
    )
    train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = UNet().to(device)
    pos_weight = compute_pos_weight(dataset, train_indices, device)
    criterion = SegmentationLoss(pos_weight=pos_weight, bce_weight=args.bce_weight)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    category_dir = Path(args.output_root) / args.category
    ensure_dir(category_dir)

    history_rows: list[dict] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_bce_losses: list[float] = []
    val_bce_losses: list[float] = []
    train_dice_losses: list[float] = []
    val_dice_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        progress_desc = f"Epoch {epoch + 1}/{args.epochs}"
        for _ in tqdm(range(1), desc=progress_desc, leave=False):
            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                train=True,
                grad_clip=args.grad_clip,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                train=False,
            )

        scheduler.step(val_stats.total_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_losses.append(train_stats.total_loss)
        val_losses.append(val_stats.total_loss)
        train_bce_losses.append(train_stats.bce_loss)
        val_bce_losses.append(val_stats.bce_loss)
        train_dice_losses.append(train_stats.dice_loss)
        val_dice_losses.append(val_stats.dice_loss)

        improved = val_stats.total_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_stats.total_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            model_path = Path("models") / f"unet_{args.category}.pth"
            ensure_dir(model_path.parent)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

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

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_stats.total_loss:.4f}, "
            f"val_loss={val_stats.total_loss:.4f}, "
            f"val_bce={val_stats.bce_loss:.4f}, "
            f"val_dice={val_stats.dice_loss:.4f}, "
            f"lr={current_lr:.6f}"
        )

        task.get_logger().report_scalar("Loss", "train", train_stats.total_loss, epoch)
        task.get_logger().report_scalar("Loss", "validation", val_stats.total_loss, epoch)
        task.get_logger().report_scalar("Validation Error", args.category, val_stats.total_loss, epoch)
        task.get_logger().report_scalar("BCE", "train", train_stats.bce_loss, epoch)
        task.get_logger().report_scalar("BCE", "validation", val_stats.bce_loss, epoch)
        task.get_logger().report_scalar("Dice loss", "train", train_stats.dice_loss, epoch)
        task.get_logger().report_scalar("Dice loss", "validation", val_stats.dice_loss, epoch)
        task.get_logger().report_scalar("Learning rate", "optimizer", current_lr, epoch)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}: "
                f"validation loss has not improved for {args.patience} epochs."
            )
            break

    model_path = Path("models") / f"unet_{args.category}.pth"

    save_csv(history_rows, category_dir / "training_history.csv")
    save_training_curve(train_losses, val_losses, category_dir / "training_curve.png")
    save_validation_error_curve(val_losses, category_dir / "validation_error_curve.png", best_epoch=best_epoch)
    save_loss_components_curve(
        train_bce_losses,
        val_bce_losses,
        train_dice_losses,
        val_dice_losses,
        category_dir / "loss_components_curve.png",
    )

    save_csv(dataset.summary_rows(), category_dir / "dataset_summary.csv")
    save_csv(subset_rows(dataset, train_indices), category_dir / "train_split.csv")
    save_csv(subset_rows(dataset, val_indices), category_dir / "val_split.csv")

    split_info = {
        "category": args.category,
        "image_size": args.image_size,
        "seed": args.seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "pos_weight": round(float(pos_weight.item()), 6),
        "bce_weight": args.bce_weight,
        "dice_weight": round(1.0 - args.bce_weight, 6),
        "grad_clip": args.grad_clip,
        "notes": (
            "MVTec AD has masks only for the test split, so this educational "
            "project trains U-Net on a local split of labeled test images."
        ),
    }
    with (category_dir / "split_info.json").open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=False)

    history_path = category_dir / "training_history.csv"
    curve_path = category_dir / "training_curve.png"
    validation_curve_path = category_dir / "validation_error_curve.png"
    components_curve_path = category_dir / "loss_components_curve.png"
    dataset_summary_path = category_dir / "dataset_summary.csv"
    train_split_path = category_dir / "train_split.csv"
    val_split_path = category_dir / "val_split.csv"
    split_info_path = category_dir / "split_info.json"

    report_csv_table(task, "Training History", "loss", history_path)
    report_csv_table(task, "Dataset Summary", "dataset", dataset_summary_path)
    report_csv_table(task, "Train Split", "dataset", train_split_path)
    report_csv_table(task, "Validation Split", "dataset", val_split_path)
    report_image(task, "Training Curve", "plots", curve_path)
    report_image(task, "Validation Error Curve", "plots", validation_curve_path)
    report_image(task, "Loss Components", "plots", components_curve_path)

    upload_if_exists(task, "model_weights", model_path)
    upload_if_exists(task, "training_history_csv", history_path)
    upload_if_exists(task, "dataset_summary_csv", dataset_summary_path)
    upload_if_exists(task, "train_split_csv", train_split_path)
    upload_if_exists(task, "val_split_csv", val_split_path)
    upload_if_exists(task, "split_info_json", split_info_path)
    upload_if_exists(task, "training_curve_png", curve_path)
    upload_if_exists(task, "validation_error_curve_png", validation_curve_path)
    upload_if_exists(task, "loss_components_curve_png", components_curve_path)

    print(f"Model saved to {model_path}")
    print(f"Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.4f}")
    print(f"Artifacts saved to {category_dir}")
    task.close()


if __name__ == "__main__":
    main()
