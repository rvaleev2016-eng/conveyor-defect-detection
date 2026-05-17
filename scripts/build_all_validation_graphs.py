import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.data.segmentation_dataset import MVTecSegmentationDataset
from src.training_utils import compute_pos_weight, run_epoch, SegmentationLoss
from src.visualization import ensure_dir


@dataclass
class TrainResult:
    history_rows: list[dict]
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild validation-error graphs for all main categories")
    parser.add_argument("--categories", default="bottle,capsule,metal_nut,pill")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs_per_category", type=int, default=12)
    parser.add_argument("--epochs_all_in_one", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--bce_weight", type=float, default=0.6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_root", default="results/teacher_revision/all_validation_graphs")
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


def plot_validation_error_ru(history_rows: list[dict], output_path: Path, title: str) -> tuple[int, float]:
    epochs = [int(row["epoch"]) for row in history_rows]
    validation_error = [float(row["validation_error"]) for row in history_rows]
    best_epoch = min(range(len(validation_error)), key=lambda idx: validation_error[idx]) + 1
    best_value = validation_error[best_epoch - 1]

    plt.figure(figsize=(8.6, 5.1))
    plt.plot(epochs, validation_error, marker="o", linewidth=2.2, color="#c0392b", label="Validation error")
    plt.scatter([best_epoch], [best_value], color="#1f77b4", s=95, zorder=5, label="Лучшая эпоха")
    plt.axvline(best_epoch, color="#1f77b4", linestyle="--", alpha=0.7)
    plt.annotate(
        f"Лучшая эпоха = {best_epoch}\nValidation error = {best_value:.6f}",
        xy=(best_epoch, best_value),
        xytext=(best_epoch + 0.5, best_value + max(0.04, best_value * 0.08)),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=10,
    )
    plt.title(title)
    plt.xlabel("Эпохи")
    plt.ylabel("Validation error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return best_epoch, best_value


def write_explanation_ru(output_path: Path, name: str, best_epoch: int, best_value: float, stopped_epoch: int) -> None:
    lines = [
        f"# Обоснование графика validation error: {name}",
        "",
        "## Что показывает график",
        "- по горизонтали отложены эпохи обучения",
        "- по вертикали отложена validation error",
        "- каждая точка показывает качество модели на валидации после завершения очередной эпохи",
        "",
        "## Какой результат получен",
        f"- лучшая эпоха: `{best_epoch}`",
        f"- минимальная validation error: `{best_value:.6f}`",
        f"- всего фактически выполнено эпох: `{stopped_epoch}`",
        "",
        "## Как это объяснять",
        "- пока validation error уменьшается, модель становится лучше на новых данных",
        "- после лучшей эпохи важно смотреть, появляется ли новый устойчивый минимум",
        "- если новый устойчивый минимум не появляется, дальнейшее обучение не даёт надёжного улучшения",
        "",
        "## Короткий вывод для защиты",
        "Количество эпох выбрано по validation error: лучшая модель определяется не последней эпохой, "
        "а эпохой с минимальной ошибкой на валидации.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_single_model(
    name: str,
    train_dataset,
    val_dataset,
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
) -> TrainResult:
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
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stopped_epoch = 0

    for epoch in range(epochs):
        stopped_epoch = epoch + 1
        for _ in tqdm(range(1), desc=f"{name} | epoch {epoch + 1}/{epochs}", leave=False):
            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                train=True,
                grad_clip=grad_clip,
            )
            val_stats = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

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
        )

        print(
            f"{name}: epoch {epoch + 1}/{epochs}, "
            f"train_loss={train_stats.total_loss:.4f}, "
            f"val_loss={val_stats.total_loss:.4f}, "
            f"lr={current_lr:.6f}"
        )

        if stale_epochs >= patience:
            print(f"{name}: early stopping at epoch {epoch + 1}")
            break

    return TrainResult(
        history_rows=history_rows,
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 6),
        stopped_epoch=stopped_epoch,
    )


def main() -> None:
    args = parse_args()
    device = get_device()
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    category_data: dict[str, dict] = {}
    for offset, category in enumerate(categories):
        dataset = MVTecSegmentationDataset(args.data_root, category, image_size=args.image_size)
        train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed + offset)
        category_data[category] = {
            "dataset": dataset,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }

    summary_rows: list[dict] = []

    for category, info in category_data.items():
        result_dir = output_root / category
        ensure_dir(result_dir)
        result = train_single_model(
            name=category,
            train_dataset=Subset(info["dataset"], info["train_indices"]),
            val_dataset=Subset(info["dataset"], info["val_indices"]),
            pos_weight=compute_pos_weight(info["dataset"], info["train_indices"], device),
            epochs=args.epochs_per_category,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            bce_weight=args.bce_weight,
            grad_clip=args.grad_clip,
            device=device,
        )
        save_csv(result.history_rows, result_dir / "training_history.csv")
        best_epoch, best_value = plot_validation_error_ru(
            result.history_rows,
            result_dir / "validation_error_epochs_ru.png",
            title=f"{category}: зависимость validation error от числа эпох",
        )
        write_explanation_ru(
            result_dir / "validation_error_epochs_explanation_ru.md",
            category,
            best_epoch,
            best_value,
            result.stopped_epoch,
        )
        summary_rows.append(
            {
                "model_name": category,
                "training_mode": "per-category",
                "best_epoch": best_epoch,
                "best_validation_error": round(best_value, 6),
                "stopped_epoch": result.stopped_epoch,
            }
        )

    combined_train_subsets = []
    combined_val_subsets = []
    combined_positive_weight_indices: list[tuple[MVTecSegmentationDataset, int]] = []
    for info in category_data.values():
        combined_train_subsets.append(Subset(info["dataset"], info["train_indices"]))
        combined_val_subsets.append(Subset(info["dataset"], info["val_indices"]))
        combined_positive_weight_indices.extend((info["dataset"], index) for index in info["train_indices"])

    total_positive = 0.0
    total_negative = 0.0
    for dataset, index in combined_positive_weight_indices:
        mask = dataset[index]["mask"]
        total_positive += float(mask.sum().item())
        total_negative += float(mask.numel() - mask.sum().item())
    all_pos_weight = torch.tensor(total_negative / max(total_positive, 1.0), dtype=torch.float32, device=device)

    all_result_dir = output_root / "all_in_one"
    ensure_dir(all_result_dir)
    all_result = train_single_model(
        name="all_in_one",
        train_dataset=ConcatDataset(combined_train_subsets),
        val_dataset=ConcatDataset(combined_val_subsets),
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
    )
    save_csv(all_result.history_rows, all_result_dir / "training_history.csv")
    best_epoch, best_value = plot_validation_error_ru(
        all_result.history_rows,
        all_result_dir / "validation_error_epochs_ru.png",
        title="all-in-one: зависимость validation error от числа эпох",
    )
    write_explanation_ru(
        all_result_dir / "validation_error_epochs_explanation_ru.md",
        "all-in-one",
        best_epoch,
        best_value,
        all_result.stopped_epoch,
    )
    summary_rows.append(
        {
            "model_name": "all_in_one",
            "training_mode": "all-in-one",
            "best_epoch": best_epoch,
            "best_validation_error": round(best_value, 6),
            "stopped_epoch": all_result.stopped_epoch,
        }
    )

    save_csv(summary_rows, output_root / "validation_graphs_summary.csv")
    print(f"Saved validation graphs to {output_root}")


if __name__ == "__main__":
    main()
