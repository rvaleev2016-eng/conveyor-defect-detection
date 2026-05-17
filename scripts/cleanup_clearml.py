import csv
from pathlib import Path

from clearml import Task


PROJECT_NAME = "Conveyor Defect Detection"


TASK_COMMENTS = {
    "3cdf1f611deb4b3abde17a1c4b85e354": (
        "Обучение U-Net только на категории bottle. "
        "Здесь показаны основные графики обучения: ошибка на обучении, validation error "
        "и сохранение лучшей модели по валидации."
    ),
    "f6ae942b2e534d8f8de27822f5cd7c02": (
        "Обучение U-Net только на категории capsule. "
        "Задача показывает, как модель обучается на одной категории дефектов и как меняется validation error по эпохам."
    ),
    "2592450c033d4f37a22356accc8c7bfb": (
        "Обучение U-Net только на категории metal_nut. "
        "В задаче оставлены только основные материалы: графики обучения, история ошибок и лучшая модель."
    ),
    "686921a36c51424bb7467d504e735c24": (
        "Обучение U-Net только на категории pill. "
        "Цель задачи показать обучение модели на одной категории и выбор лучшей эпохи по validation error."
    ),
    "14f9e37090e549079b753c6d09f4c65d": (
        "Оценка модели, обученной отдельно на категории bottle. "
        "Здесь собраны итоговые метрики, визуал дефектов и масок, а также примеры наложения предсказания на изображение."
    ),
    "de8f156d4f684435a6064a578b1fae25": (
        "Оценка модели, обученной отдельно на категории capsule. "
        "В задаче есть основные метрики сегментации и визуальные примеры: изображение, истинная маска и предсказанная маска."
    ),
    "d1a36abb62ec4c70ac727d746d4469c5": (
        "Оценка модели, обученной отдельно на категории metal_nut. "
        "Используется для наглядного показа качества сегментации на одной категории дефектов."
    ),
    "8038c97f18a34517aca557c43b264e36": (
        "Оценка модели, обученной отдельно на категории pill. "
        "В задаче собраны итоговые метрики и визуальные примеры сегментации дефектов."
    ),
    "918fb86b08d04067803661a4b95eb295": (
        "Общее обучение одной U-Net сразу на нескольких категориях: bottle, capsule, metal_nut и pill. "
        "Здесь видно, как ведёт себя единая модель и как меняется validation error при совместном обучении."
    ),
    "40fed20972db4a5f9cc4089b322412ee": (
        "Оценка общей модели на категории bottle. "
        "Нужна для сравнения общей модели с моделью, обученной отдельно на одной категории."
    ),
    "3c5e0d1580434f1ab5275f48c28ef1c1": (
        "Оценка общей модели на категории capsule. "
        "В задаче отражены основные метрики и визуальные примеры работы общей модели."
    ),
    "693e06d543a241798412800cd0186662": (
        "Оценка общей модели на категории metal_nut. "
        "Используется для сравнения качества общего обучения и отдельного обучения по категории."
    ),
    "8bd08f3b841f478abf83e600e73ae90f": (
        "Оценка общей модели на категории pill. "
        "Здесь можно сравнить итоговую сегментацию общей модели с отдельным обучением по категории."
    ),
    "c337cfab185a4f64b2b46cabbcbdeb98": (
        "Сводная таблица итоговых результатов проекта. "
        "В ней оставлены только основные показатели: категория, режим обучения, Dice и IoU. "
        "Задача нужна для быстрого сравнения отдельного обучения по категориям и общего обучения."
    ),
}


DELETE_TASK_IDS = {
    "41242a4c30e24091b19d1c1ab20e8987",
    "ea3356ba2acb4db2b68554111478a294",
    "5ea38f9185fa4e11971e3fd8e2e9b442",
    "3c8c7f74f5634f83bcae060be6b55130",
    "7e82614d898a475cac2d6994343f2c36",
}


def write_clean_summary(source_path: Path, target_path: Path) -> None:
    with source_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                "category": row["category"],
                "training_mode": "отдельно" if row["model_type"] == "per-category" else "общее",
                "dice": row["dice"],
                "iou": row["iou"],
            }
            for row in reader
        ]

    with target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "training_mode", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for task_id in sorted(DELETE_TASK_IDS):
        try:
            task = Task.get_task(task_id=task_id)
        except Exception:
            continue
        task.delete(delete_artifacts_and_models=False, skip_models_used_by_other_tasks=True)
        print(f"Deleted task {task_id}: {task.name}")

    for task_id, comment in TASK_COMMENTS.items():
        task = Task.get_task(task_id=task_id)
        task.set_comment(comment)
        task.close()
        print(f"Updated comment for {task_id}: {task.name}")

    comparison_dir = Path("results/final/comparison")
    source_csv = comparison_dir / "final_comparison.csv"
    clean_csv = comparison_dir / "final_summary.csv"
    write_clean_summary(source_csv, clean_csv)
    print(f"Saved clean summary table to {clean_csv}")


if __name__ == "__main__":
    main()
