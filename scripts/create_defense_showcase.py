import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.clearml_utils import init_task, report_csv_table, report_image, upload_if_exists


def write_markdown(path: Path) -> None:
    text = """# Сценарий Защиты На 3-5 Минут

## 1. Постановка задачи
- Проект решает задачу сегментации дефектов на изображениях.
- На вход модель получает изображение детали, на выходе возвращает бинарную маску дефекта.
- Для решения выбрана архитектура U-Net.

## 2. Почему выбрана U-Net
- U-Net подходит для попиксельной сегментации.
- Skip-connections помогают сохранять границы дефекта.
- Модель достаточно сильная, но при этом понятная для учебной защиты.

## 3. Как устроено обучение
- Используется локальное разбиение размеченного набора на train и validation.
- Во время обучения контролируется validation error.
- Если validation error перестаёт улучшаться, learning rate уменьшается.
- Если улучшения нет несколько эпох подряд, применяется early stopping.

## 4. Почему используется BCE + Dice
- BCE отвечает за попиксельную бинарную классификацию.
- Dice усиливает совпадение формы маски.
- Такая комбинация лучше работает при сильном дисбалансе между фоном и областью дефекта.

## 5. Что показывают графики
- Training Curve показывает динамику ошибки на обучении и валидации.
- Validation Error Curve показывает, где находится лучшая эпоха и когда начинается ухудшение.
- Loss Components Curve показывает вклад BCE и Dice loss.

## 6. Какие метрики используются
- Dice показывает качество перекрытия предсказанной и истинной маски.
- IoU показывает пересечение к объединению масок.
- Чем выше обе метрики, тем лучше сегментация.

## 7. Какие результаты получены
- В проекте сравниваются отдельное обучение по категориям и общее обучение.
- Для категории bottle отдельное обучение лучше общего.
- Это нужно показывать по отдельной таблице bottle, а не только по общей сводке.
- Для bottle также было проведено сравнение оптимизаторов, и лучший результат дал вариант с SGD.

## 8. Как объяснять замечание про threshold
- Threshold относится к постобработке вероятностной карты.
- Это не основной аргумент про качество обучения модели.
- Его можно показать отдельно как чувствительность постобработки, но не как доказательство лучшего обучения.

## 9. Что показывают визуальные результаты
- В примерах показаны исходное изображение, истинная маска, предсказанная маска и overlay.
- Есть лучшие, средние и проблемные примеры.
- Это позволяет показать не только сильные стороны модели, но и её ограничения.

## 10. Итоговый вывод
- U-Net подходит для сегментации дефектов.
- Контроль через validation error делает обучение устойчивее.
- Для bottle отдельное обучение лучше общего.
- Лучший обучающий результат для bottle был получен после сравнения оптимизаторов и параметров.
- Threshold нужно подавать как постобработку, а не как главный итог обучения.
"""
    path.write_text(text, encoding="utf-8")


def write_showcase_tables(comparison_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    with comparison_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    bottle_rows = [row for row in rows if row["category"] == "bottle"]
    bottle_rows_sorted = sorted(bottle_rows, key=lambda row: float(row["dice"]), reverse=True)

    bottle_comparison_path = output_dir / "bottle_clear_comparison.csv"
    with bottle_comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "training_mode", "dice", "iou", "conclusion"],
        )
        writer.writeheader()
        for index, row in enumerate(bottle_rows_sorted):
            writer.writerow(
                {
                    "category": row["category"],
                    "training_mode": row["training_mode"],
                    "dice": row["dice"],
                    "iou": row["iou"],
                    "conclusion": "лучший результат для bottle" if index == 0 else "хуже лучшего результата",
                }
            )

    showcase_summary_path = output_dir / "showcase_summary.csv"
    with showcase_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "category", "training_mode", "dice", "iou", "comment"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "section": "Ключевое сравнение bottle",
                "category": bottle_rows_sorted[0]["category"],
                "training_mode": bottle_rows_sorted[0]["training_mode"],
                "dice": bottle_rows_sorted[0]["dice"],
                "iou": bottle_rows_sorted[0]["iou"],
                "comment": "Это лучший результат для bottle в сравнении отдельно против общего.",
            }
        )
        writer.writerow(
            {
                "section": "Ключевое сравнение bottle",
                "category": bottle_rows_sorted[1]["category"],
                "training_mode": bottle_rows_sorted[1]["training_mode"],
                "dice": bottle_rows_sorted[1]["dice"],
                "iou": bottle_rows_sorted[1]["iou"],
                "comment": "Это более слабый результат для bottle.",
            }
        )
        for row in rows:
            writer.writerow(
                {
                    "section": "Полная сводка",
                    "category": row["category"],
                    "training_mode": row["training_mode"],
                    "dice": row["dice"],
                    "iou": row["iou"],
                    "comment": "",
                }
            )

    return bottle_comparison_path, showcase_summary_path


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    title: str,
    columns: int = 2,
    thumb_size: tuple[int, int] = (520, 180),
    header_height: int = 70,
    padding: int = 20,
) -> None:
    if not image_paths:
        return

    rows = (len(image_paths) + columns - 1) // columns
    width = columns * thumb_size[0] + (columns + 1) * padding
    height = header_height + rows * thumb_size[1] + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 20), title, fill="black")

    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(thumb_size)
        col = index % columns
        row = index // columns
        x = padding + col * thumb_size[0]
        y = header_height + padding + row * thumb_size[1]
        canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    repo_url = "https://github.com/rvaleev2016-eng/conveyor-defect-detection"
    defense_dir = base / "results" / "final" / "defense"
    teacher_dir = base / "results" / "teacher_revision" / "sgd_bottle" / "bottle"
    quick_dir = defense_dir / "quick_pack"
    comparison_dir = base / "results" / "final" / "comparison"
    report_dir = base / "results" / "teacher_revision" / "report_data"
    graphs_dir = base / "results" / "teacher_revision" / "training_graphs"
    all_graphs_dir = base / "results" / "teacher_revision" / "all_validation_graphs"
    showcase_md = defense_dir / "defense_showcase_script.md"
    visuals_dir = defense_dir / "showcase_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(showcase_md)
    bottle_comparison_csv, showcase_summary_csv = write_showcase_tables(
        comparison_dir / "final_summary.csv",
        defense_dir,
    )

    task = init_task(
        project_name="Conveyor Defect Detection",
        task_name="Defense Showcase (U-Net)",
        tags=["final", "defense", "showcase", "unet", "segmentation"],
        params={
            "category_focus": "bottle",
            "showcase_mode": "single task for 3-5 minute defense",
            "github_repository": repo_url,
            "includes": [
                "bottle clear comparison",
                "validation epoch argument",
                "epochs vs validation error graph in russian",
                "validation error graphs for all categories",
                "optimizer comparison",
                "metric glossary",
                "project summary",
                "training curves",
                "validation error",
                "metrics",
                "threshold appendix",
                "visual collages",
                "curated examples",
                "final comparison",
            ],
        },
    )
    task.set_comment(
        "Итоговая витринная задача для защиты проекта за 3-5 минут. "
        f"GitHub-репозиторий проекта: {repo_url}. "
        "Внутри собраны все ключевые этапы в правильном порядке: постановка задачи, "
        "обучение U-Net, validation error, обоснование числа эпох, сравнение оптимизаторов, "
        "итоговые метрики, приложение по threshold, визуальные примеры сегментации и финальная сравнительная таблица. "
        "Добавлен отдельный русский график зависимости validation error от числа эпох и русское пояснение к нему для защиты. "
        "Дополнительно добавлены русские графики validation error для всех категорий и общего обучения. "
        "Для bottle добавлена отдельная явная таблица, где видно, что отдельное обучение лучше общего. "
        "Threshold вынесен в приложение и не используется как главный аргумент качества обучения. "
        "Эту задачу можно показывать преподавателю как единую точку входа без перехода по остальным задачам."
    )

    final_summary_csv = comparison_dir / "final_summary.csv"
    eval_summary_csv = teacher_dir / "evaluation_summary.csv"
    threshold_csv = quick_dir / "threshold_sweep.csv"
    validation_epoch_csv = report_dir / "validation_epoch_argument.csv"
    metric_glossary_csv = report_dir / "metric_glossary.csv"
    optimizer_argument_csv = report_dir / "optimizer_argument.csv"
    threshold_appendix_csv = report_dir / "threshold_appendix.csv"
    error_analysis_md = quick_dir / "error_analysis.md"
    quick_summary_json = quick_dir / "quick_summary.json"
    training_history_csv = teacher_dir / "training_history.csv"
    validation_graph_ru_png = graphs_dir / "validation_error_epochs_ru.png"
    validation_graph_ru_md = graphs_dir / "validation_error_epochs_explanation_ru.md"
    all_graphs_summary_csv = all_graphs_dir / "validation_graphs_summary.csv"
    github_index_md = defense_dir / "GITHUB_PROJECT_INDEX.md"

    report_csv_table(task, "1. Явное Сравнение bottle: отдельно против общего", "bottle_compare", bottle_comparison_csv)
    report_csv_table(task, "2. Обоснование Validation Error И Числа Эпох", "validation_epochs", validation_epoch_csv)
    report_csv_table(task, "3. Сравнение Оптимизаторов И Параметров", "optimizer_argument", optimizer_argument_csv)
    report_csv_table(task, "4. Сводка Для Порядка Показa", "showcase_summary", showcase_summary_csv)
    report_csv_table(task, "5. Полная Итоговая Сводная Таблица", "main_metrics", final_summary_csv)
    report_csv_table(task, "6. Метрики По Категории bottle", "bottle_metrics", eval_summary_csv)
    report_csv_table(task, "7. Словарь Всех Цифр И Показателей", "metric_glossary", metric_glossary_csv)
    report_csv_table(task, "8. Приложение: Threshold Как Постобработка", "threshold_appendix", threshold_appendix_csv)
    report_csv_table(task, "9. История Обучения bottle", "training_history", training_history_csv)
    report_image(task, "10. График: Эпохи И Validation Error", "validation_error_ru", validation_graph_ru_png)
    upload_if_exists(task, "validation_error_epochs_ru_png", validation_graph_ru_png)
    report_csv_table(task, "10a. Сводка Validation Error По Всем Моделям", "all_validation_summary", all_graphs_summary_csv)

    for image_path, title, series in [
        (teacher_dir / "training_curve.png", "11. Training Curve", "training"),
        (teacher_dir / "validation_error_curve.png", "12. Validation Error Curve", "validation"),
        (teacher_dir / "loss_components_curve.png", "13. Loss Components Curve", "loss_parts"),
    ]:
        report_image(task, title, series, image_path)
        upload_if_exists(task, image_path.stem, image_path)

    for name, title, series in [
        ("bottle", "13a. Validation Error: bottle", "all_val_bottle"),
        ("capsule", "13b. Validation Error: capsule", "all_val_capsule"),
        ("metal_nut", "13c. Validation Error: metal_nut", "all_val_metal_nut"),
        ("pill", "13d. Validation Error: pill", "all_val_pill"),
        ("all_in_one", "13e. Validation Error: all-in-one", "all_val_all_in_one"),
    ]:
        image_path = all_graphs_dir / name / "validation_error_epochs_ru.png"
        report_image(task, title, series, image_path)
        upload_if_exists(task, f"{name}_validation_error_epochs_ru_png", image_path)

    curated_root = quick_dir / "curated_examples"
    curated_all_paths: list[Path] = []
    for group_name in ["best", "medium", "worst"]:
        group_dir = curated_root / group_name
        group_paths = sorted(group_dir.glob("*.png"))
        curated_all_paths.extend(group_paths)
        for index, image_path in enumerate(group_paths, start=1):
            report_image(task, f"14. Curated Examples: {group_name}", f"{group_name}_{index}", image_path)
            upload_if_exists(task, f"{group_name}_{index}_png", image_path)

    example_dir = teacher_dir / "examples"
    full_example_paths = sorted(example_dir.glob("example_*.png"))
    for index, image_path in enumerate(full_example_paths, start=1):
        report_image(task, "15. Full Segmentation Examples", f"example_{index}", image_path)
        upload_if_exists(task, f"full_example_{index}_png", image_path)

    full_examples_sheet = visuals_dir / "full_examples_contact_sheet.png"
    curated_sheet = visuals_dir / "curated_examples_contact_sheet.png"
    create_contact_sheet(
        full_example_paths,
        full_examples_sheet,
        title="Полные примеры сегментации bottle",
        columns=2,
        thumb_size=(520, 180),
    )
    create_contact_sheet(
        curated_all_paths,
        curated_sheet,
        title="Лучшие, средние и проблемные примеры bottle",
        columns=2,
        thumb_size=(520, 180),
    )
    for image_path, title, series in [
        (full_examples_sheet, "16. Contact Sheet: Full Examples", "full_sheet"),
        (curated_sheet, "17. Contact Sheet: Curated Examples", "curated_sheet"),
    ]:
        report_image(task, title, series, image_path)
        upload_if_exists(task, image_path.stem, image_path)

    task.get_logger().report_scalar("Showcase Metrics", "bottle_final_separate_dice", 0.606644, 0)
    task.get_logger().report_scalar("Showcase Metrics", "bottle_final_general_dice", 0.1875, 0)
    task.get_logger().report_scalar("Showcase Metrics", "bottle_sgd_dice_threshold_0.5", 0.538667, 0)
    task.get_logger().report_scalar("Showcase Metrics", "bottle_sgd_iou_threshold_0.5", 0.42929, 0)
    task.get_logger().report_scalar("Showcase Metrics", "best_validation_error_sgd", 0.380891, 0)
    task.get_logger().report_scalar("Showcase Metrics", "best_validation_error_adamw", 0.402278, 0)
    task.get_logger().report_scalar("Showcase Metrics", "threshold_best_postprocess", 0.7, 0)

    for artifact_name, artifact_path in [
        ("bottle_clear_comparison_csv", bottle_comparison_csv),
        ("showcase_summary_csv", showcase_summary_csv),
        ("final_summary_csv", final_summary_csv),
        ("evaluation_summary_csv", eval_summary_csv),
        ("threshold_sweep_csv", threshold_csv),
        ("validation_epoch_argument_csv", validation_epoch_csv),
        ("metric_glossary_csv", metric_glossary_csv),
        ("optimizer_argument_csv", optimizer_argument_csv),
        ("threshold_appendix_csv", threshold_appendix_csv),
        ("training_history_csv", training_history_csv),
        ("validation_error_epochs_explanation_ru_md", validation_graph_ru_md),
        ("all_validation_graphs_summary_csv", all_graphs_summary_csv),
        ("github_project_index_md", github_index_md),
        ("error_analysis_md", error_analysis_md),
        ("quick_summary_json", quick_summary_json),
        ("defense_showcase_script_md", showcase_md),
        ("full_examples_contact_sheet_png", full_examples_sheet),
        ("curated_examples_contact_sheet_png", curated_sheet),
        ("clearml_defense_guide_docx", defense_dir / "clearml_defense_guide.docx"),
        ("full_project_analysis_report_docx", defense_dir / "full_project_analysis_report.docx"),
    ]:
        upload_if_exists(task, artifact_name, artifact_path)

    for name in ["bottle", "capsule", "metal_nut", "pill", "all_in_one"]:
        upload_if_exists(
            task,
            f"{name}_validation_error_epochs_explanation_ru_md",
            all_graphs_dir / name / "validation_error_epochs_explanation_ru.md",
        )
        upload_if_exists(
            task,
            f"{name}_training_history_csv",
            all_graphs_dir / name / "training_history.csv",
        )

    print(f"Defense showcase task id: {task.id}")
    task.close()


if __name__ == "__main__":
    main()
