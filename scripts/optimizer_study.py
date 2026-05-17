import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_csv_table, upload_if_exists
from src.data.segmentation_dataset import MVTecSegmentationDataset
from src.training_utils import SegmentationLoss, compute_pos_weight, run_epoch
from src.visualization import ensure_dir


@dataclass
class Variant:
    name: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float = 0.9


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
    return train_indices, val_indices


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    category = "bottle"
    data_root = "data/raw"
    image_size = 256
    batch_size = 4
    epochs = 20
    patience = 5
    min_delta = 1e-4
    seed = 42
    val_ratio = 0.2
    bce_weight = 0.6
    grad_clip = 1.0

    variants = [
        Variant(name="adamw_lr5e-4", optimizer="adamw", lr=5e-4, weight_decay=1e-4),
        Variant(name="adam_lr5e-4", optimizer="adam", lr=5e-4, weight_decay=1e-4),
        Variant(name="rmsprop_lr1e-4", optimizer="rmsprop", lr=1e-4, weight_decay=1e-4),
        Variant(name="sgd_lr1e-2", optimizer="sgd", lr=1e-2, weight_decay=1e-4),
        Variant(name="adamw_lr1e-3", optimizer="adamw", lr=1e-3, weight_decay=1e-4),
    ]

    task = init_task(
        project_name="Conveyor Defect Detection",
        task_name="Optimizer Study (bottle)",
        tags=["study", "optimizer", "ablation", "bottle", "unet"],
        params={
            "category": category,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": min_delta,
            "seed": seed,
            "val_ratio": val_ratio,
            "bce_weight": bce_weight,
            "grad_clip": grad_clip,
        },
    )
    task.set_comment(
        "Сравнительное исследование оптимизаторов и параметров обучения для категории bottle. "
        "Цель задачи — показать, насколько сильно можно улучшить validation error за счёт выбора оптимизатора и базовых гиперпараметров."
    )

    output_dir = Path("results/teacher_revision/optimizer_study")
    ensure_dir(output_dir)

    device = get_device()
    dataset = MVTecSegmentationDataset(root=data_root, category=category, image_size=image_size)
    train_indices, val_indices = split_indices(len(dataset), val_ratio, seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    pos_weight = compute_pos_weight(dataset, train_indices, device)

    summary_rows: list[dict] = []
    history_rows: list[dict] = []

    for variant in variants:
        model = UNet().to(device)
        loss_fn = SegmentationLoss(pos_weight=pos_weight, bce_weight=bce_weight)
        optimizer = build_optimizer(model, variant)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(epochs):
            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                train=True,
                grad_clip=grad_clip,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                optimizer,
                loss_fn,
                device,
                train=False,
            )
            scheduler.step(val_stats.total_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            improved = val_stats.total_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_stats.total_loss
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1

            history_rows.append(
                {
                    "variant": variant.name,
                    "epoch": epoch + 1,
                    "train_loss": round(train_stats.total_loss, 6),
                    "validation_error": round(val_stats.total_loss, 6),
                    "learning_rate": f"{current_lr:.8f}",
                    "best_so_far": "yes" if improved else "no",
                }
            )

            task.get_logger().report_scalar("Validation Error by Variant", variant.name, val_stats.total_loss, epoch)
            task.get_logger().report_scalar("Train Loss by Variant", variant.name, train_stats.total_loss, epoch)

            if stale_epochs >= patience:
                break

        summary_rows.append(
            {
                "variant": variant.name,
                "optimizer": variant.optimizer,
                "lr": variant.lr,
                "weight_decay": variant.weight_decay,
                "best_epoch": best_epoch,
                "best_validation_error": round(best_val_loss, 6),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: float(row["best_validation_error"]))
    best_error = float(summary_rows[0]["best_validation_error"])
    for row in summary_rows:
        row["gap_to_best"] = round(float(row["best_validation_error"]) - best_error, 6)

    summary_csv = output_dir / "optimizer_study_summary.csv"
    history_csv = output_dir / "optimizer_study_history.csv"
    save_csv(summary_rows, summary_csv)
    save_csv(history_rows, history_csv)

    conclusion = {
        "category": category,
        "best_variant": summary_rows[0]["variant"],
        "best_validation_error": summary_rows[0]["best_validation_error"],
        "worst_validation_error": summary_rows[-1]["best_validation_error"],
        "best_to_worst_gap": round(
            float(summary_rows[-1]["best_validation_error"]) - float(summary_rows[0]["best_validation_error"]),
            6,
        ),
        "note": (
            "Если разрыв между лучшим и соседними вариантами небольшой, "
            "это означает, что базовое качество уже устойчиво и радикально лучше получить трудно."
        ),
    }
    conclusion_path = output_dir / "optimizer_study_conclusion.json"
    conclusion_path.write_text(json.dumps(conclusion, indent=2, ensure_ascii=False), encoding="utf-8")

    report_csv_table(task, "Optimizer Study Summary", "study", summary_csv)
    report_csv_table(task, "Optimizer Study History", "study", history_csv)
    upload_if_exists(task, "optimizer_study_summary_csv", summary_csv)
    upload_if_exists(task, "optimizer_study_history_csv", history_csv)
    upload_if_exists(task, "optimizer_study_conclusion_json", conclusion_path)

    print(f"Optimizer study complete. Best variant: {summary_rows[0]['variant']}")
    print(f"Results saved to {output_dir}")
    task.close()


if __name__ == "__main__":
    main()
