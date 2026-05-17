import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_validation_error(history_rows: list[dict], output_path: Path) -> tuple[int, float]:
    epochs = [int(row["epoch"]) for row in history_rows]
    validation_error = [float(row["validation_error"]) for row in history_rows]
    best_epoch = min(range(len(validation_error)), key=lambda idx: validation_error[idx]) + 1
    best_value = validation_error[best_epoch - 1]

    plt.figure(figsize=(8, 4.8))
    plt.plot(epochs, validation_error, marker="o", linewidth=2, color="#c0392b", label="Validation error")
    plt.scatter([best_epoch], [best_value], color="#1f77b4", s=90, zorder=5, label="Best epoch")
    plt.axvline(best_epoch, color="#1f77b4", linestyle="--", alpha=0.7)
    plt.annotate(
        f"best epoch = {best_epoch}\nval_error = {best_value:.6f}",
        xy=(best_epoch, best_value),
        xytext=(best_epoch + 0.6, best_value + 0.1),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=10,
    )
    plt.title("Validation Error by Epoch (bottle, final model)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return best_epoch, best_value


def plot_validation_error_ru(history_rows: list[dict], output_path: Path) -> tuple[int, float]:
    epochs = [int(row["epoch"]) for row in history_rows]
    validation_error = [float(row["validation_error"]) for row in history_rows]
    best_epoch = min(range(len(validation_error)), key=lambda idx: validation_error[idx]) + 1
    best_value = validation_error[best_epoch - 1]

    plt.figure(figsize=(8.4, 5.0))
    plt.plot(
        epochs,
        validation_error,
        marker="o",
        linewidth=2.2,
        color="#c0392b",
        label="Validation error",
    )
    plt.scatter([best_epoch], [best_value], color="#1f77b4", s=90, zorder=5, label="Лучшая эпоха")
    plt.axvline(best_epoch, color="#1f77b4", linestyle="--", alpha=0.7)
    plt.annotate(
        f"Лучшая эпоха = {best_epoch}\nValidation error = {best_value:.6f}",
        xy=(best_epoch, best_value),
        xytext=(best_epoch + 0.7, best_value + 0.12),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=10,
    )
    plt.title("Зависимость validation error от числа эпох")
    plt.xlabel("Эпохи")
    plt.ylabel("Validation error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return best_epoch, best_value


def plot_epoch_decision(history_rows: list[dict], output_path: Path, best_epoch: int) -> None:
    epochs = [int(row["epoch"]) for row in history_rows]
    validation_error = [float(row["validation_error"]) for row in history_rows]
    learning_rate = [float(row["learning_rate"]) for row in history_rows]

    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    ax1.plot(epochs, validation_error, marker="o", linewidth=2, color="#c0392b", label="Validation error")
    ax1.axvline(best_epoch, color="#1f77b4", linestyle="--", alpha=0.7, label="Chosen stop region")
    ax1.axvspan(best_epoch, epochs[-1], color="#1f77b4", alpha=0.08)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation error", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rate, marker="s", linewidth=1.8, color="#2c7fb8", label="Learning rate")
    ax2.set_ylabel("Learning rate", color="#2c7fb8")
    ax2.tick_params(axis="y", labelcolor="#2c7fb8")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.title("Why the chosen number of epochs is justified")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_optimizer_comparison(optimizer_rows: list[dict], output_path: Path) -> None:
    labels = [row["optimizer"] for row in optimizer_rows]
    values = [float(row["best_validation_error"]) for row in optimizer_rows]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values, color=["#2c7fb8", "#7fcdbb", "#41b6c4", "#fdae61", "#d7191c"][: len(labels)])
    plt.title("Comparison of optimizers by best validation error")
    plt.xlabel("Optimizer")
    plt.ylabel("Best validation error")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_explanation(
    output_path: Path,
    best_epoch: int,
    best_value: float,
    optimizer_rows: list[dict],
    split_info: dict,
) -> None:
    best_optimizer = optimizer_rows[0]
    lines = [
        "# Обоснование Графиков Обучения",
        "",
        "## 1. График validation error",
        f"- лучшая эпоха: `{best_epoch}`",
        f"- минимальная validation error на финальной bottle-модели: `{best_value:.6f}`",
        "- смысл графика: показать, на какой эпохе модель дала лучший результат на validation",
        "- после лучшей эпохи устойчивого улучшения уже нет, поэтому ранняя остановка обоснована",
        "",
        "## 2. График обоснования числа эпох",
        f"- в `split_info.json` зафиксировано: best_epoch = `{split_info['best_epoch']}`, best_val_loss = `{split_info['best_val_loss']}`",
        "- на графике одновременно показаны validation error и learning rate",
        "- это позволяет объяснить, что после ухудшения validation error learning rate был снижен, но новый устойчивый минимум уже не появился",
        "",
        "## 3. График сравнения оптимизаторов",
        f"- лучший вариант по исследованию: `{best_optimizer['variant']}`",
        f"- его best_validation_error: `{best_optimizer['best_validation_error']}`",
        "- смысл графика: показать преподавателю, что были проверены разные оптимизаторы и гиперпараметры",
        "- если значения близки, это аргумент в пользу устойчивости результата; если один вариант заметно лучше, значит именно его и нужно брать как финальный",
        "",
        "## 4. Как этим пользоваться на защите",
        "- сначала показать график validation error и сказать, почему лучшая эпоха выбрана корректно",
        "- затем показать график с learning rate и обосновать раннюю остановку",
        "- после этого показать сравнение оптимизаторов и объяснить, почему финальная конфигурация выбрана осознанно",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_explanation_ru(output_path: Path, best_epoch: int, best_value: float, split_info: dict) -> None:
    lines = [
        "# Обоснование графика validation error",
        "",
        "## Что показывает график",
        "- по горизонтали отложены эпохи обучения",
        "- по вертикали отложена validation error",
        "- каждая точка показывает, какой была ошибка модели на валидации после завершения очередной эпохи",
        "",
        "## Как читать этот график",
        f"- лучшая эпоха на графике: `{best_epoch}`",
        f"- минимальная validation error: `{best_value:.6f}`",
        "- пока validation error уменьшается, модель улучшает качество на новых данных",
        "- когда validation error перестаёт устойчиво уменьшаться, это означает, что дальнейшее обучение уже не даёт надёжного улучшения",
        "",
        "## Почему число эпох выбрано правильно",
        f"- в итоговой модели зафиксировано: best_epoch = `{split_info['best_epoch']}`",
        f"- лучшая validation error в split_info.json: `{split_info['best_val_loss']}`",
        "- после лучшей эпохи на графике нет нового устойчивого минимума, поэтому продолжать обучение дальше не было смысла",
        "- именно поэтому модель выбирается по лучшей validation error, а не по последней эпохе",
        "",
        "## Короткий вывод для защиты",
        "Количество эпох выбрано корректно, потому что лучшая validation error достигнута на лучшей эпохе, "
        "а после неё устойчивого улучшения уже не наблюдается.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    history_rows = read_csv(base / "results" / "teacher_revision" / "sgd_bottle" / "bottle" / "training_history.csv")
    optimizer_rows = read_csv(base / "results" / "teacher_revision" / "report_data" / "optimizer_argument.csv")
    split_info = json.loads(
        (base / "results" / "teacher_revision" / "sgd_bottle" / "bottle" / "split_info.json").read_text(encoding="utf-8")
    )

    output_dir = base / "results" / "teacher_revision" / "training_graphs"
    ensure_dir(output_dir)

    best_epoch, best_value = plot_validation_error(history_rows, output_dir / "validation_error_argument.png")
    plot_validation_error_ru(history_rows, output_dir / "validation_error_epochs_ru.png")
    plot_epoch_decision(history_rows, output_dir / "epoch_decision_argument.png", best_epoch=best_epoch)
    plot_optimizer_comparison(optimizer_rows, output_dir / "optimizer_comparison_argument.png")
    write_explanation(
        output_dir / "training_graphs_explanation.md",
        best_epoch=best_epoch,
        best_value=best_value,
        optimizer_rows=optimizer_rows,
        split_info=split_info,
    )
    write_explanation_ru(
        output_dir / "validation_error_epochs_explanation_ru.md",
        best_epoch=best_epoch,
        best_value=best_value,
        split_info=split_info,
    )
    print(f"Training graphs saved to {output_dir}")


if __name__ == "__main__":
    main()
