"""Segmentation losses and label-preserving augmentation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss, ``1 - mean_c Dice_c`` over the batch.

    Dice is computed per class over all pixels of the batch, so small classes weigh as much as background --
    this directly targets (m)IoU-like overlap, complementing pixel-wise cross-entropy.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.float().softmax(dim=1)  # float32 even under autocast
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = target != self.ignore_index
        target = torch.where(valid, target, torch.zeros_like(target))
        one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs, one_hot = probs * valid, one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """``CE`` or ``CE + dice_weight * Dice``."""

    def __init__(self, num_classes: int, kind: str = "ce", dice_weight: float = 1.0) -> None:
        super().__init__()
        if kind not in ("ce", "ce_dice"):
            raise ValueError(f"loss must be 'ce' or 'ce_dice', got {kind!r}")
        self.kind = kind
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss(num_classes) if kind == "ce_dice" else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits.float(), target)
        if self.dice is not None:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss


@torch.no_grad()
def random_flip_rot90(images: torch.Tensor, masks: torch.Tensor,
                      generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random element of the dihedral group D4 (8 flips/rotations), applied on-device.

    All shape classes are invariant under these maps (a mirrored crown is still a crown, a rotated pentagon is
    still a pentagon), so labels are preserved exactly and no interpolation is involved. Requires square images
    for 90-degree rotations.
    """
    b = images.shape[0]
    ks = torch.randint(0, 4, (b,), generator=generator).tolist()
    flips = torch.randint(0, 2, (b,), generator=generator).tolist()
    out_i, out_m = [], []
    for i in range(b):
        img, msk = images[i], masks[i]
        if flips[i]:
            img, msk = img.flip(-1), msk.flip(-1)
        if ks[i]:
            img, msk = torch.rot90(img, ks[i], dims=(-2, -1)), torch.rot90(msk, ks[i], dims=(-2, -1))
        out_i.append(img)
        out_m.append(msk)
    return torch.stack(out_i), torch.stack(out_m)
