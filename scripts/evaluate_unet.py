import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_csv_table, report_image, upload_if_exists
from src.data.segmentation_dataset import MVTecSegmentationDataset
from src.visualization import ensure_dir, save_prediction_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on the saved validation split")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--example_count", type=int, default=12)
    parser.add_argument("--output_root", default="results")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


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
        task_name=f"Evaluate U-Net ({args.category})",
        tags=["unet", "segmentation", "evaluation", args.category],
        params=vars(args),
    )

    category_dir = Path(args.output_root) / args.category
    split_path = category_dir / "split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}. Run training first.")

    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    val_indices = split_info["val_indices"]

    dataset = MVTecSegmentationDataset(
        root=args.data_root,
        category=args.category,
        image_size=args.image_size,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)

    model_path = Path("models") / f"unet_{args.category}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics_rows: list[dict] = []
    dice_total = 0.0
    iou_total = 0.0
    saved_examples = 0
    examples_dir = category_dir / "examples"
    ensure_dir(examples_dir)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > args.threshold).float()

            batch_dice = dice_score(predictions, masks).item()
            batch_iou = iou_score(predictions, masks).item()
            dice_total += batch_dice
            iou_total += batch_iou

            for idx in range(images.size(0)):
                image_name = Path(batch["image_path"][idx]).name
                defect_type = batch["defect_type"][idx]
                pixel_ratio = float(predictions[idx].mean().item())
                metrics_rows.append(
                    {
                        "image_name": image_name,
                        "defect_type": defect_type,
                        "predicted_defect_ratio": round(pixel_ratio, 6),
                    }
                )

                if saved_examples < args.example_count:
                    save_prediction_figure(
                        image=images[idx].cpu(),
                        target_mask=masks[idx].cpu(),
                        predicted_mask=predictions[idx].cpu(),
                        path=examples_dir / f"example_{saved_examples + 1:02d}.png",
                        title=f"{args.category}: {image_name}",
                    )
                    saved_examples += 1

    summary_rows = [
        {
            "metric": "Dice",
            "value": round(dice_total / len(val_loader), 6),
            "explanation": "Overlap between predicted mask and ground-truth mask. Closer to 1 is better.",
        },
        {
            "metric": "IoU",
            "value": round(iou_total / len(val_loader), 6),
            "explanation": "Intersection over Union for segmentation masks. Closer to 1 is better.",
        },
        {
            "metric": "Threshold",
            "value": args.threshold,
            "explanation": "Probability cutoff used to convert the predicted map into a binary mask.",
        },
    ]

    save_csv(summary_rows, category_dir / "evaluation_summary.csv")
    save_csv(metrics_rows, category_dir / "prediction_table.csv")

    summary_path = category_dir / "evaluation_summary.csv"
    prediction_path = category_dir / "prediction_table.csv"
    report_csv_table(task, "Evaluation Summary", "metrics", summary_path)
    report_csv_table(task, "Prediction Table", "predictions", prediction_path)

    for row in summary_rows:
        if row["metric"] in {"Dice", "IoU"}:
            task.get_logger().report_scalar(
                title="Segmentation Metrics",
                series=row["metric"],
                value=float(row["value"]),
                iteration=0,
            )

    example_paths = sorted(examples_dir.glob("example_*.png"))
    for index, example_path in enumerate(example_paths):
        report_image(task, "Segmentation Examples", f"example_{index + 1}", example_path)

    upload_if_exists(task, "evaluation_summary_csv", summary_path)
    upload_if_exists(task, "prediction_table_csv", prediction_path)
    for index, example_path in enumerate(example_paths):
        upload_if_exists(task, f"example_{index + 1}_png", example_path)

    print(f"Evaluation finished for category '{args.category}'")
    for row in summary_rows:
        print(f"{row['metric']}: {row['value']}")
    print(f"Saved tables and examples to {category_dir}")
    task.close()


if __name__ == "__main__":
    main()
