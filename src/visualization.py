from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_numpy(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().squeeze().numpy()
    else:
        array = np.asarray(mask).squeeze()
    return array.astype(np.float32)


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def save_training_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker="o", linewidth=2, label="Train loss")
    plt.plot(epochs, val_losses, marker="s", linewidth=2, label="Validation loss")
    plt.title("Train and Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_validation_error_curve(val_losses: list[float], path: Path, best_epoch: int | None = None) -> None:
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(val_losses) + 1)
    plt.plot(epochs, val_losses, marker="s", linewidth=2, color="#c0392b", label="Validation error")
    if best_epoch is not None and 1 <= best_epoch <= len(val_losses):
        best_value = val_losses[best_epoch - 1]
        plt.scatter([best_epoch], [best_value], color="#1f77b4", s=70, zorder=5, label="Best epoch")
        plt.axvline(best_epoch, color="#1f77b4", linestyle="--", alpha=0.6)
    plt.title("Validation Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_loss_components_curve(
    train_bce: list[float],
    val_bce: list[float],
    train_dice: list[float],
    val_dice: list[float],
    path: Path,
) -> None:
    epochs = range(1, len(train_bce) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_bce, marker="o", linewidth=2, label="Train BCE")
    axes[0].plot(epochs, val_bce, marker="s", linewidth=2, label="Validation BCE")
    axes[0].set_title("BCE Component")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_dice, marker="o", linewidth=2, label="Train Dice loss")
    axes[1].plot(epochs, val_dice, marker="s", linewidth=2, label="Validation Dice loss")
    axes[1].set_title("Dice Loss Component")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_comparison_plot(rows: list[dict], path: Path) -> None:
    labels = [f"{row['category']}\n{row['model_type']}" for row in rows]
    dice_values = [float(row["dice"]) for row in rows]
    iou_values = [float(row["iou"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    ax.bar(x - width / 2, dice_values, width=width, label="Dice")
    ax.bar(x + width / 2, iou_values, width=width, label="IoU")
    ax.set_title("Final Comparison: Per-category vs All-in-one")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prediction_figure(
    image: torch.Tensor,
    target_mask: torch.Tensor | np.ndarray | None,
    predicted_mask: torch.Tensor | np.ndarray,
    path: Path,
    title: str,
) -> None:
    image_np = image_to_numpy(image)
    predicted_np = mask_to_numpy(predicted_mask)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input image")
    if target_mask is None:
        axes[1].text(0.5, 0.5, "No ground truth", ha="center", va="center", fontsize=12)
        axes[1].set_title("Ground-truth mask")
    else:
        target_np = mask_to_numpy(target_mask)
        axes[1].imshow(target_np, cmap="gray")
        axes[1].set_title("Ground-truth mask")
    axes[2].imshow(predicted_np, cmap="gray")
    axes[2].set_title("Predicted mask")
    axes[3].imshow(image_np)
    axes[3].imshow(predicted_np, cmap="jet", alpha=0.45)
    axes[3].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
