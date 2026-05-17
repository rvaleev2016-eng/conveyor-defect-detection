import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    teacher_dir = base / "results" / "teacher_review" / "bottle"
    quick_dir = base / "results" / "quick_improvements" / "bottle"
    study_dir = base / "results" / "teacher_revision" / "optimizer_study"
    output_dir = base / "results" / "teacher_revision" / "report_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = read_csv(teacher_dir / "training_history.csv")
    eval_rows = read_csv(teacher_dir / "evaluation_summary.csv")
    threshold_rows = read_csv(quick_dir / "threshold_sweep.csv")
    optimizer_rows = []
    if (study_dir / "optimizer_study_summary.csv").exists():
        optimizer_rows = read_csv(study_dir / "optimizer_study_summary.csv")
    elif (base / "results" / "teacher_revision" / "optimizer_study_local" / "optimizer_study_local_summary.csv").exists():
        optimizer_rows = read_csv(base / "results" / "teacher_revision" / "optimizer_study_local" / "optimizer_study_local_summary.csv")
    split_info = json.loads((teacher_dir / "split_info.json").read_text(encoding="utf-8"))

    best_epoch = split_info["best_epoch"]
    best_val_loss = split_info["best_val_loss"]

    epoch_rows = []
    for row in training_rows:
        epoch_rows.append(
            {
                "epoch": row["epoch"],
                "validation_error": row["validation_error"],
                "learning_rate": row["learning_rate"],
                "best_model": row["best_model"],
                "comment": (
                    "лучшая эпоха"
                    if int(row["epoch"]) == int(best_epoch)
                    else "после лучшей эпохи ошибка уже не улучшилась"
                    if int(row["epoch"]) > int(best_epoch)
                    else "модель ещё улучшалась"
                ),
            }
        )

    metric_rows = [
        {
            "term": "train_loss / train_error",
            "meaning": "общая ошибка на обучающей выборке",
            "why_important": "показывает, насколько модель подстроилась под train-данные",
        },
        {
            "term": "validation_error / val_loss",
            "meaning": "ошибка на validation выборке",
            "why_important": "главный показатель для выбора лучшей модели и контроля переобучения",
        },
        {
            "term": "train_bce / val_bce",
            "meaning": "попиксельная бинарная ошибка",
            "why_important": "показывает качество классификации каждого пикселя как дефект или фон",
        },
        {
            "term": "train_dice_loss / val_dice_loss",
            "meaning": "ошибка совпадения формы маски",
            "why_important": "показывает, насколько хорошо модель восстанавливает саму область дефекта",
        },
        {
            "term": "learning_rate",
            "meaning": "скорость изменения весов модели",
            "why_important": "уменьшается при ухудшении validation error и стабилизирует обучение",
        },
        {
            "term": "best_epoch",
            "meaning": "эпоха с минимальной validation error",
            "why_important": "именно эта версия модели считается лучшей, а не последняя эпоха",
        },
        {
            "term": "Dice",
            "meaning": "степень совпадения предсказанной и истинной маски",
            "why_important": "основная метрика сегментации, чем ближе к 1, тем лучше",
        },
        {
            "term": "IoU",
            "meaning": "отношение пересечения к объединению масок",
            "why_important": "более строгая метрика качества сегментации",
        },
        {
            "term": "threshold",
            "meaning": "порог перевода вероятностей в бинарную маску",
            "why_important": "это постобработка, а не доказательство качества обучения",
        },
        {
            "term": "pos_weight",
            "meaning": "вес положительного класса в BCE",
            "why_important": "компенсирует дисбаланс между маленькой областью дефекта и большим фоном",
        },
    ]

    threshold_note_rows = []
    for row in threshold_rows:
        threshold_note_rows.append(
            {
                "threshold": row["threshold"],
                "dice": row["dice"],
                "iou": row["iou"],
                "comment": "это чувствительность постобработки, а не новое обучение модели",
            }
        )

    teacher_notes = {
        "validation_epoch_argument": {
            "best_epoch": best_epoch,
            "best_validation_error": best_val_loss,
            "statement": (
                "Количество эпох выбрано корректно, потому что лучшая validation error достигнута на "
                f"{best_epoch}-й эпохе, а дальше устойчивого улучшения уже нет."
            ),
        },
        "threshold_argument": {
            "statement": (
                "Threshold относится к постобработке вероятностной карты и не должен подаваться как "
                "главное доказательство качества обучения модели."
            )
        },
        "optimizer_argument": {
            "statement": (
                "Сравнение оптимизаторов и базовых гиперпараметров нужно использовать как аргумент, "
                "что радикально лучшее качество получить трудно, если лучшие и соседние варианты близки."
            )
        },
    }

    save_csv(epoch_rows, output_dir / "validation_epoch_argument.csv")
    save_csv(metric_rows, output_dir / "metric_glossary.csv")
    save_csv(threshold_note_rows, output_dir / "threshold_appendix.csv")
    if optimizer_rows:
        save_csv(optimizer_rows, output_dir / "optimizer_argument.csv")
    with (output_dir / "teacher_notes.json").open("w", encoding="utf-8") as handle:
        json.dump(teacher_notes, handle, indent=2, ensure_ascii=False)

    print(f"Teacher revision report data saved to {output_dir}")


if __name__ == "__main__":
    main()
