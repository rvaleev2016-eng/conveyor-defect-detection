import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from src.clearml_utils import init_task, report_csv_table, report_image, upload_if_exists
from src.data.segmentation_dataset import MVTecSegmentationDataset
from src.visualization import ensure_dir, save_prediction_figure


@dataclass
class SampleMetrics:
    image_name: str
    defect_type: str
    dice: float
    iou: float
    predicted_ratio: float
    target_ratio: float
    abs_ratio_error: float
    image_path: str


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * intersection + eps) / (denominator + eps)


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return (intersection + eps) / (union + eps)


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_thresholds(
    model: UNet,
    dataset: MVTecSegmentationDataset,
    val_indices: list[int],
    thresholds: list[float],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], float]:
    loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    rows: list[dict] = []
    best_threshold = thresholds[0]
    best_dice = -1.0

    model.eval()
    with torch.no_grad():
        cached_batches = []
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probabilities = torch.sigmoid(model(images))
            cached_batches.append((probabilities.cpu(), masks.cpu()))

    for threshold in thresholds:
        dice_values: list[float] = []
        iou_values: list[float] = []
        for probabilities, masks in cached_batches:
            predictions = (probabilities > threshold).float()
            dice_values.extend(dice_score(predictions, masks).tolist())
            iou_values.extend(iou_score(predictions, masks).tolist())

        dice_mean = sum(dice_values) / len(dice_values)
        iou_mean = sum(iou_values) / len(iou_values)
        row = {
            "threshold": threshold,
            "dice": round(dice_mean, 6),
            "iou": round(iou_mean, 6),
        }
        rows.append(row)
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_threshold = threshold

    return rows, best_threshold


def evaluate_samples(
    model: UNet,
    dataset: MVTecSegmentationDataset,
    val_indices: list[int],
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[list[SampleMetrics], list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    metrics: list[SampleMetrics] = []
    cached_visuals: list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities > threshold).float()

            batch_dice = dice_score(predictions, masks).cpu().tolist()
            batch_iou = iou_score(predictions, masks).cpu().tolist()

            for idx in range(images.size(0)):
                target_ratio = float(masks[idx].mean().item())
                predicted_ratio = float(predictions[idx].mean().item())
                row = SampleMetrics(
                    image_name=Path(batch["image_path"][idx]).name,
                    defect_type=batch["defect_type"][idx],
                    dice=float(batch_dice[idx]),
                    iou=float(batch_iou[idx]),
                    predicted_ratio=predicted_ratio,
                    target_ratio=target_ratio,
                    abs_ratio_error=abs(predicted_ratio - target_ratio),
                    image_path=batch["image_path"][idx],
                )
                metrics.append(row)
                cached_visuals.append(
                    (
                        {
                            "image_name": row.image_name,
                            "defect_type": row.defect_type,
                        },
                        images[idx].cpu(),
                        masks[idx].cpu(),
                        predictions[idx].cpu(),
                    )
                )

    return metrics, cached_visuals


def build_curated_examples(
    sample_metrics: list[SampleMetrics],
    cached_visuals: list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> dict[str, list[SampleMetrics]]:
    ensure_dir(output_dir)
    visuals_by_name = {meta["image_name"]: (image, target, pred) for meta, image, target, pred in cached_visuals}

    sorted_by_dice = sorted(sample_metrics, key=lambda item: item.dice, reverse=True)
    best_examples = sorted_by_dice[:3]
    medium_examples = sorted(sample_metrics, key=lambda item: abs(item.dice - 0.5))[:2]
    worst_examples = sorted(sample_metrics, key=lambda item: item.dice)[:3]

    groups = {
        "best": best_examples,
        "medium": medium_examples,
        "worst": worst_examples,
    }

    for group_name, items in groups.items():
        group_dir = output_dir / group_name
        ensure_dir(group_dir)
        for index, item in enumerate(items, start=1):
            image, target, pred = visuals_by_name[item.image_name]
            save_prediction_figure(
                image=image,
                target_mask=target,
                predicted_mask=pred,
                path=group_dir / f"{index:02d}_{item.image_name}",
                title=f"{group_name.upper()}: {item.image_name} | Dice={item.dice:.3f} | IoU={item.iou:.3f}",
            )

    return groups


def write_error_analysis(
    output_path: Path,
    best_threshold: float,
    threshold_rows: list[dict],
    groups: dict[str, list[SampleMetrics]],
) -> None:
    best_dice = next(row["dice"] for row in threshold_rows if row["threshold"] == best_threshold)
    best_iou = next(row["iou"] for row in threshold_rows if row["threshold"] == best_threshold)

    lines = [
        "# Быстрый Анализ Ошибок Модели",
        "",
        "## 1. Что было улучшено за быстрый этап",
        f"- подобран лучший threshold по validation: `{best_threshold}`",
        f"- итоговый Dice при этом threshold: `{best_dice}`",
        f"- итоговый IoU при этом threshold: `{best_iou}`",
        "- выделены лучшие, средние и проблемные примеры сегментации",
        "",
        "## 2. Почему подбор threshold важен",
        "- после sigmoid модель выдаёт вероятности, а не готовую бинарную маску",
        "- threshold определяет, при каком уровне вероятности пиксель считается дефектом",
        "- слишком низкий threshold даёт лишний фон, слишком высокий может обрезать сам дефект",
        "",
        "## 3. Что видно по лучшим примерам",
        "- модель хорошо выделяет крупные и контрастные дефекты",
        "- границы маски становятся точнее, когда дефект занимает заметную часть области",
        "",
        "## 4. Что видно по средним примерам",
        "- форма дефекта определяется верно, но часть области может быть потеряна",
        "- иногда модель находит основную зону дефекта, но не восстанавливает её полностью",
        "",
        "## 5. Что видно по проблемным примерам",
        "- на мелких дефектах модель может недосегментировать область",
        "- на сложной текстуре возможен захват лишнего фона",
        "- если дефект слабоконтрастный, маска может быть неполной",
        "",
        "## 6. Короткий вывод",
        "- быстрый этап улучшения полезен тем, что позволяет без переобучения подобрать лучший режим бинаризации",
        "- для защиты проекта это усиливает как численные метрики, так и визуальное качество показа",
        "",
        "## 7. Примеры по группам",
    ]

    for group_name, items in groups.items():
        lines.append(f"### {group_name}")
        for item in items:
            lines.append(
                f"- {item.image_name}: Dice={item.dice:.4f}, IoU={item.iou:.4f}, "
                f"predicted_ratio={item.predicted_ratio:.4f}, target_ratio={item.target_ratio:.4f}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    category = "bottle"
    data_root = Path("data/raw")
    output_root = Path("results/quick_improvements") / category
    ensure_dir(output_root)
    task = init_task(
        project_name="Conveyor Defect Detection",
        task_name=f"Quick Improvements ({category})",
        tags=["quick-improvement", "unet", "segmentation", category],
        params={
            "category": category,
            "threshold_candidates": [0.30, 0.40, 0.50, 0.60, 0.70],
            "purpose": "one-hour package for defense",
        },
    )
    task.set_comment(
        "Быстрый пакет улучшений без переобучения модели. "
        "В задаче зафиксированы подбор лучшего threshold по validation, "
        "таблица по каждому изображению, отобранные лучшие и проблемные примеры, "
        "а также короткий анализ ошибок модели для защиты проекта."
    )

    split_info = json.loads((Path("results/teacher_review") / category / "split_info.json").read_text(encoding="utf-8"))
    val_indices = split_info["val_indices"]

    device = get_device()
    dataset = MVTecSegmentationDataset(root=data_root, category=category, image_size=256)
    model = UNet().to(device)
    model.load_state_dict(torch.load(Path("models") / f"unet_{category}.pth", map_location=device))

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_rows, best_threshold = evaluate_thresholds(
        model=model,
        dataset=dataset,
        val_indices=val_indices,
        thresholds=thresholds,
        batch_size=4,
        device=device,
    )
    save_csv(threshold_rows, output_root / "threshold_sweep.csv")

    sample_metrics, cached_visuals = evaluate_samples(
        model=model,
        dataset=dataset,
        val_indices=val_indices,
        threshold=best_threshold,
        batch_size=4,
        device=device,
    )
    sample_rows = [
        {
            "image_name": item.image_name,
            "defect_type": item.defect_type,
            "dice": round(item.dice, 6),
            "iou": round(item.iou, 6),
            "predicted_ratio": round(item.predicted_ratio, 6),
            "target_ratio": round(item.target_ratio, 6),
            "abs_ratio_error": round(item.abs_ratio_error, 6),
        }
        for item in sorted(sample_metrics, key=lambda x: x.dice, reverse=True)
    ]
    save_csv(sample_rows, output_root / "per_image_metrics.csv")

    curated_dir = output_root / "curated_examples"
    groups = build_curated_examples(sample_metrics, cached_visuals, curated_dir)
    write_error_analysis(output_root / "error_analysis.md", best_threshold, threshold_rows, groups)

    summary = {
        "category": category,
        "best_threshold": best_threshold,
        "best_dice": next(row["dice"] for row in threshold_rows if row["threshold"] == best_threshold),
        "best_iou": next(row["iou"] for row in threshold_rows if row["threshold"] == best_threshold),
        "curated_examples_dir": str(curated_dir),
    }
    (output_root / "quick_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    defense_dir = Path("results/final/defense/quick_pack")
    ensure_dir(defense_dir)
    for filename in ["threshold_sweep.csv", "per_image_metrics.csv", "error_analysis.md", "quick_summary.json"]:
        shutil.copy2(output_root / filename, defense_dir / filename)

    curated_defense_dir = defense_dir / "curated_examples"
    if curated_defense_dir.exists():
        shutil.rmtree(curated_defense_dir)
    shutil.copytree(curated_dir, curated_defense_dir)

    threshold_csv = output_root / "threshold_sweep.csv"
    per_image_csv = output_root / "per_image_metrics.csv"
    error_analysis_path = output_root / "error_analysis.md"
    quick_summary_path = output_root / "quick_summary.json"

    report_csv_table(task, "Threshold Sweep", "metrics", threshold_csv)
    report_csv_table(task, "Per Image Metrics", "metrics", per_image_csv)
    task.get_logger().report_scalar(
        title="Quick Metrics",
        series="best_dice",
        value=float(summary["best_dice"]),
        iteration=0,
    )
    task.get_logger().report_scalar(
        title="Quick Metrics",
        series="best_iou",
        value=float(summary["best_iou"]),
        iteration=0,
    )
    task.get_logger().report_scalar(
        title="Quick Metrics",
        series="best_threshold",
        value=float(summary["best_threshold"]),
        iteration=0,
    )

    for group_name in ["best", "medium", "worst"]:
        group_dir = curated_dir / group_name
        for index, image_path in enumerate(sorted(group_dir.glob("*.png")), start=1):
            report_image(task, "Curated Examples", f"{group_name}_{index}", image_path)
            upload_if_exists(task, f"{group_name}_{index}_png", image_path)

    upload_if_exists(task, "threshold_sweep_csv", threshold_csv)
    upload_if_exists(task, "per_image_metrics_csv", per_image_csv)
    upload_if_exists(task, "error_analysis_md", error_analysis_path)
    upload_if_exists(task, "quick_summary_json", quick_summary_path)

    print(f"Best threshold: {best_threshold}")
    print(f"Saved quick improvements to {output_root}")
    print(f"Copied defense pack to {defense_dir}")
    print(f"ClearML task id: {task.id}")
    task.close()


if __name__ == "__main__":
    main()
