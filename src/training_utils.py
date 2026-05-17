from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, bce_weight: float = 0.6) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss.detach(), dice_loss.detach()


@dataclass
class EpochStats:
    total_loss: float
    bce_loss: float
    dice_loss: float


def compute_pos_weight(dataset, indices: list[int], device: torch.device) -> torch.Tensor:
    positive_pixels = 0.0
    negative_pixels = 0.0

    for index in indices:
        sample = dataset[index]
        mask = sample["mask"]
        positive_pixels += float(mask.sum().item())
        negative_pixels += float(mask.numel() - mask.sum().item())

    if positive_pixels <= 0:
        return torch.tensor(1.0, device=device)

    weight = negative_pixels / max(positive_pixels, 1.0)
    return torch.tensor(weight, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    loss_fn: SegmentationLoss,
    device: torch.device,
    train: bool,
    grad_clip: float | None = None,
) -> EpochStats:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss, bce_loss, dice_loss = loss_fn(logits, masks)

            if train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_dice += float(dice_loss.item())

    batch_count = max(len(loader), 1)
    return EpochStats(
        total_loss=total_loss / batch_count,
        bce_loss=total_bce / batch_count,
        dice_loss=total_dice / batch_count,
    )
