import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.data.segmentation_dataset import MVTecSegmentationDataset
from src.training_utils import SegmentationLoss, compute_pos_weight, run_epoch


@dataclass
class Variant:
    name: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float = 0.9


def get_device() -> torch.device:
    return torch.device("cpu")


def split_indices(length: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(length * val_ratio))
    return indices[val_size:], indices[:val_size]


def build_optimizer(model: UNet, variant: Variant):
    if variant.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    if variant.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    if variant.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=variant.lr,
            weight_decay=variant.weight_decay,
            momentum=variant.momentum,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=variant.lr,
        weight_decay=variant.weight_decay,
        momentum=variant.momentum,
    )


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    category = "bottle"
    device = get_device()
    dataset = MVTecSegmentationDataset("data/raw", category=category, image_size=256)
    train_indices, val_indices = split_indices(len(dataset), 0.2, 42)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=4, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=4, shuffle=False)
    pos_weight = compute_pos_weight(dataset, train_indices, device)

    variants = [
        Variant(name="adamw_lr5e-4", optimizer="adamw", lr=5e-4, weight_decay=1e-4),
        Variant(name="adam_lr5e-4", optimizer="adam", lr=5e-4, weight_decay=1e-4),
        Variant(name="rmsprop_lr1e-4", optimizer="rmsprop", lr=1e-4, weight_decay=1e-4),
    ]

    output_dir = Path("results/teacher_revision/optimizer_study_local")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for variant in variants:
        model = UNet().to(device)
        optimizer = build_optimizer(model, variant)
        loss_fn = SegmentationLoss(pos_weight=pos_weight, bce_weight=0.6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        best_val = float("inf")
        best_epoch = 0
        stale = 0

        print(f"Running {variant.name} on {device}")
        for epoch in range(8):
            train_stats = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True, grad_clip=1.0)
            val_stats = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
            scheduler.step(val_stats.total_loss)
            if val_stats.total_loss < best_val - 1e-4:
                best_val = val_stats.total_loss
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            print(
                f"  epoch {epoch + 1}: train_loss={train_stats.total_loss:.4f}, "
                f"val_error={val_stats.total_loss:.4f}"
            )
            if stale >= 3:
                break

        rows.append(
            {
                "variant": variant.name,
                "optimizer": variant.optimizer,
                "lr": variant.lr,
                "best_epoch": best_epoch,
                "best_validation_error": round(best_val, 6),
            }
        )
        print(variant.name, best_epoch, round(best_val, 6))

    rows = sorted(rows, key=lambda x: float(x["best_validation_error"]))
    best = float(rows[0]["best_validation_error"])
    for row in rows:
        row["gap_to_best"] = round(float(row["best_validation_error"]) - best, 6)
    save_csv(rows, output_dir / "optimizer_study_local_summary.csv")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
