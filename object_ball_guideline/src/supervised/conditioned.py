from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.supervised.model import ConvBlock, DownBlock, UpBlock


class ConditionedGuidelineUNet(nn.Module):
    """UNet with a candidate/object-ball conditioning channel and validity head."""

    def __init__(self, base_channels: int = 32, in_channels: int = 4) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.stem = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        self.up3 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)
        self.validity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c5, c2),
            nn.ReLU(inplace=True),
            nn.Linear(c2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        bottleneck = self.down4(s3)
        decoded = self.up3(bottleneck, s3)
        decoded = self.up2(decoded, s2)
        decoded = self.up1(decoded, s1)
        decoded = self.up0(decoded, s0)
        return {
            "mask_logits": self.head(decoded),
            "validity_logits": self.validity_head(bottleneck),
        }


class ConditionedGuidelineLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        validity_weight: float = 0.35,
        line_weight: float = 2.0,
        neighborhood_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.validity_weight = float(validity_weight)
        self.line_weight = float(line_weight)
        self.neighborhood_weight = float(neighborhood_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        validity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_logits = outputs["mask_logits"]
        validity_logits = outputs["validity_logits"]
        pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32, device=targets.device)
        dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
        weights = 1.0 + self.line_weight * targets + self.neighborhood_weight * (dilated - targets).clamp_min(0.0)
        bce = F.binary_cross_entropy_with_logits(mask_logits, targets, pos_weight=pos_weight, weight=weights)
        probs = torch.sigmoid(mask_logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        line = ((probs - targets).abs() * weights).mean()
        validity_loss = F.binary_cross_entropy_with_logits(validity_logits, validity)
        total = bce + dice + 0.2 * line + self.validity_weight * validity_loss
        return {
            "loss": total,
            "mask_bce": bce.detach(),
            "dice_loss": dice.detach(),
            "line_loss": line.detach(),
            "validity_loss": validity_loss.detach(),
        }


def conditioned_dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2.0 * intersection + 1.0) / (union + 1.0)
