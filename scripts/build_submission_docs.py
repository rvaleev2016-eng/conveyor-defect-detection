import csv
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_URL = "https://github.com/rvaleev2016-eng/conveyor-defect-detection"
CLEARML_URL = "https://app.clear.ml/projects/1b01d7ca2ca741fb881df31ee748a1e6/experiments/0db5742cb68f4c46919dcb64fed478e7/output/log"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_asset(source: Path, target: Path) -> None:
    ensure_dir(target.parent)
    shutil.copy2(source, target)


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def build_docs(base: Path) -> None:
    docs_dir = base / "docs"
    assets_dir = docs_dir / "assets"
    ensure_dir(assets_dir)

    final_summary = read_csv(base / "results" / "final" / "comparison" / "final_summary.csv")
    validation_summary = read_csv(
        base / "results" / "teacher_revision" / "all_validation_graphs" / "validation_graphs_summary.csv"
    )
    bottle_eval = read_csv(base / "results" / "teacher_revision" / "sgd_bottle" / "bottle" / "evaluation_summary.csv")

    assets = {
        "validation_bottle.png": base
        / "results"
        / "teacher_revision"
        / "training_graphs"
        / "validation_error_epochs_ru.png",
        "validation_all_models_bottle.png": base
        / "results"
        / "teacher_revision"
        / "all_validation_graphs"
        / "bottle"
        / "validation_error_epochs_ru.png",
        "validation_all_models_capsule.png": base
        / "results"
        / "teacher_revision"
        / "all_validation_graphs"
        / "capsule"
        / "validation_error_epochs_ru.png",
        "validation_all_models_metal_nut.png": base
        / "results"
        / "teacher_revision"
        / "all_validation_graphs"
        / "metal_nut"
        / "validation_error_epochs_ru.png",
        "validation_all_models_pill.png": base
        / "results"
        / "teacher_revision"
        / "all_validation_graphs"
        / "pill"
        / "validation_error_epochs_ru.png",
        "validation_all_models_all_in_one.png": base
        / "results"
        / "teacher_revision"
        / "all_validation_graphs"
        / "all_in_one"
        / "validation_error_epochs_ru.png",
        "curated_examples_contact_sheet.png": base
        / "results"
        / "final"
        / "defense"
        / "showcase_visuals"
        / "curated_examples_contact_sheet.png",
        "full_examples_contact_sheet.png": base
        / "results"
        / "final"
        / "defense"
        / "showcase_visuals"
        / "full_examples_contact_sheet.png",
    }
    for target_name, source_path in assets.items():
        copy_asset(source_path, assets_dir / target_name)

    result_rows = [["Категория", "Режим", "Dice", "IoU"]]
    for row in final_summary:
        result_rows.append([row["category"], row["training_mode"], row["dice"], row["iou"]])

    validation_rows = [["Модель", "Режим", "Лучшая эпоха", "Лучшая validation error", "Остановка"]]
    for row in validation_summary:
        validation_rows.append(
            [
                row["model_name"],
                row["training_mode"],
                row["best_epoch"],
                row["best_validation_error"],
                row["stopped_epoch"],
            ]
        )

    bottle_metrics_rows = [["Метрика", "Значение"]]
    for row in bottle_eval:
        bottle_metrics_rows.append([row["metric"], row["value"]])

    (docs_dir / "README.md").write_text(
        "\n".join(
            [
                "# Материалы Для Сдачи",
                "",
                f"- GitHub-репозиторий: {REPO_URL}",
                f"- Главная задача ClearML: {CLEARML_URL}",
                "",
                "## Что находится в папке",
                "",
                "- `report.pdf` — итоговый PDF-отчёт для сдачи.",
                "- `report_source.md` — текстовая основа отчёта.",
                "- `task_journal.md` — журнал выполненной работы по основным этапам.",
                "- `assets/` — ключевые графики и визуалы для отчёта.",
                "",
                "## Что открыть в первую очередь",
                "",
                "1. `report.pdf`",
                "2. `task_journal.md`",
                "3. `assets/validation_bottle.png`",
                "4. ClearML showcase-задачу",
            ]
        ),
        encoding="utf-8",
    )

    (docs_dir / "task_journal.md").write_text(
        "\n".join(
            [
                "# Журнал Задач",
                "",
                "> Рабочая итоговая версия журнала по основным этапам подготовки проекта.",
                "> Если преподаватель требует журнал по всем неделям семестра, этот файл можно дополнить ранними учебными записями.",
                "",
                "| Дата | Выполненная работа |",
                "|---|---|",
                "| 2026-05-11 | Проведён разбор проекта, удалены лишние ветки, оставлена сегментация на базе U-Net. |",
                "| 2026-05-11 | Исправлены ошибки в train/eval-скриптах, путях к данным и зависимостях. |",
                "| 2026-05-11 | Добавлены ClearML-логи, визуализация результатов и единая showcase-задача. |",
                "| 2026-05-11 | Подготовлены Word-отчёты и сценарий защиты. |",
                "| 2026-05-11 | Добавлены графики validation error и аргументация по лучшей эпохе. |",
                "| 2026-05-11 | Проведено сравнение оптимизаторов и обновлена bottle-модель. |",
                "| 2026-05-11 | Добавлены графики validation error по всем категориям и all-in-one. |",
                "| 2026-05-17 | Подготовлена структура `/docs`, собраны материалы для GitHub и создан `report.pdf`. |",
            ]
        ),
        encoding="utf-8",
    )

    (docs_dir / "report_source.md").write_text(
        "\n".join(
            [
                "# Итоговый Отчёт По Проекту",
                "",
                "## 1. Тема проекта",
                "",
                "Сегментация дефектов на изображениях с помощью U-Net на датасете MVTec AD.",
                "",
                "## 2. Цель",
                "",
                "Построить модель, которая не только определяет наличие дефекта, но и выделяет его область в виде бинарной маски.",
                "",
                "## 3. Реализация",
                "",
                "- модель: `U-Net`;",
                "- функция потерь: `0.6 * BCEWithLogitsLoss + 0.4 * DiceLoss`;",
                "- контроль качества: `validation error`, `Dice`, `IoU`;",
                "- защита от переобучения: `ReduceLROnPlateau` и `early stopping`;",
                "- логирование и витрина проекта: `ClearML`.",
                "",
                "## 4. Основные результаты",
                "",
                markdown_table(result_rows),
                "",
                "## 5. Validation error по моделям",
                "",
                markdown_table(validation_rows),
                "",
                "## 6. Основные выводы",
                "",
                "- для категории `bottle` отдельное обучение показало лучший результат, чем общее;",
                "- лучшая bottle-модель выбиралась по минимальной `validation error`, а не по последней эпохе;",
                "- графики `validation error` подготовлены для всех основных категорий и общей модели;",
                "- финальная презентация проекта собрана в одной showcase-задаче ClearML.",
                "",
                "## 7. Ссылки",
                "",
                f"- GitHub: {REPO_URL}",
                f"- ClearML showcase: {CLEARML_URL}",
            ]
        ),
        encoding="utf-8",
    )

    build_pdf(docs_dir, result_rows, validation_rows, bottle_metrics_rows)


def build_pdf(docs_dir: Path, result_rows: list[list[str]], validation_rows: list[list[str]], bottle_metrics_rows: list[list[str]]) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="BodyRu", fontSize=11, leading=15))

    report_path = docs_dir / "report.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    story.append(Paragraph("Итоговый отчёт по проекту Conveyor Defect Detection", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Курс: «Прикладные методы искусственного интеллекта»", styles["BodyRu"]))
    story.append(Paragraph(f"GitHub: {REPO_URL}", styles["BodyRu"]))
    story.append(Paragraph(f"ClearML Showcase: {CLEARML_URL}", styles["BodyRu"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Постановка задачи", styles["Heading2"]))
    story.append(
        Paragraph(
            "Проект решает задачу сегментации дефектов на изображениях деталей. "
            "На вход модель получает изображение, а на выходе формирует бинарную маску области дефекта.",
            styles["BodyRu"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("2. Почему выбрана U-Net", styles["Heading2"]))
    story.append(
        Paragraph(
            "U-Net выбрана как базовая архитектура для попиксельной сегментации: она хорошо сохраняет пространственную структуру "
            "изображения и даёт наглядный результат, удобный для учебной защиты.",
            styles["BodyRu"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3. Основные результаты", styles["Heading2"]))
    story.append(make_table(result_rows, col_widths=[4.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            "Ключевой вывод: для bottle отдельное обучение оказалось сильнее общего режима по метрикам Dice и IoU.",
            styles["BodyRu"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4. Bottle: финальные метрики улучшенной модели", styles["Heading2"]))
    story.append(make_table(bottle_metrics_rows, col_widths=[6.5 * cm, 4.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5. Validation error и выбор числа эпох", styles["Heading2"]))
    story.append(
        Paragraph(
            "Лучшая эпоха выбиралась по минимуму validation error. После прекращения устойчивого улучшения применялись снижение "
            "learning rate и ранняя остановка обучения.",
            styles["BodyRu"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(str(docs_dir / "assets" / "validation_bottle.png"), width=16.5 * cm, height=9.8 * cm))
    story.append(Spacer(1, 0.25 * cm))
    story.append(make_table(validation_rows, col_widths=[3.3 * cm, 3.0 * cm, 3.0 * cm, 4.2 * cm, 2.2 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("6. Визуальные результаты сегментации", styles["Heading2"]))
    story.append(
        Paragraph(
            "Для защиты подготовлены контактные листы с лучшими, средними и проблемными примерами сегментации.",
            styles["BodyRu"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(docs_dir / "assets" / "curated_examples_contact_sheet.png"), width=16.5 * cm, height=12.0 * cm))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(docs_dir / "assets" / "full_examples_contact_sheet.png"), width=16.5 * cm, height=12.0 * cm))

    story.append(PageBreak())
    story.append(Paragraph("7. Графики validation error по всем моделям", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    for image_name in [
        "validation_all_models_bottle.png",
        "validation_all_models_capsule.png",
        "validation_all_models_metal_nut.png",
        "validation_all_models_pill.png",
        "validation_all_models_all_in_one.png",
    ]:
        story.append(Image(str(docs_dir / "assets" / image_name), width=15.8 * cm, height=9.2 * cm))
        story.append(Spacer(1, 0.18 * cm))

    story.append(PageBreak())
    story.append(Paragraph("8. Итоговый вывод", styles["Heading2"]))
    story.append(
        Paragraph(
            "Проект приведён к воспроизводимой и сдаваемой структуре: код организован по модулям, README содержит инструкции по запуску "
            "и основные результаты, дополнительные материалы вынесены в /docs, а итоговая демонстрация собрана в одной ClearML showcase-задаче. "
            "Таким образом, репозиторий готов к очной защите и показу через GitHub.",
            styles["BodyRu"],
        )
    )

    doc.build(story)


def make_table(rows: list[list[str]], col_widths: list[float]) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
            ]
        )
    )
    return table


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    build_docs(base)
    print(f"Submission docs created in {base / 'docs'}")


if __name__ == "__main__":
    main()
