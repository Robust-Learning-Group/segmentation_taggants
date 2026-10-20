"""Confusion-matrix based evaluation metrics for semantic segmentation.

All metrics are derived from a single ``num_classes x num_classes`` confusion
matrix accumulated over the *entire* evaluation set (rows = ground truth,
columns = prediction). This is the standard protocol (e.g. PASCAL VOC,
Cityscapes, mmsegmentation): IoU is computed from dataset-level TP/FP/FN
counts, **not** averaged per batch or per image, so results do not depend on
batch size.

Metrics reported:

* ``pixel_acc``          -- overall pixel accuracy, sum(TP) / #valid pixels.
* ``mean_acc``           -- mean per-class recall (a.k.a. mean class accuracy).
* ``miou``               -- mean IoU over all classes present in GT or prediction.
* ``miou_fg``            -- mean IoU excluding the background class. Background
  dominates synthetic shape images, so this is the more informative number.
* ``fwiou``              -- frequency-weighted IoU (weighted by GT pixel share).
* ``mdice``              -- mean Dice / F1 score over classes.
* ``per_class_*``        -- IoU, Dice, recall (acc) and precision per class.

Classes that appear neither in the ground truth nor in the predictions have an
undefined IoU; they are reported as ``NaN`` and excluded from the means
(``nanmean``), matching common practice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SegmentationMetrics:
    """Streaming confusion-matrix accumulator for semantic segmentation.

    Example:
        >>> metric = SegmentationMetrics(num_classes=8)
        >>> for images, masks in loader:
        ...     metric.update(model(images), masks)   # logits or label maps
        >>> results = metric.compute()
        >>> results["miou"]
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        background_index: Optional[int] = 0,
        class_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        """Initialize the accumulator.

        Args:
            num_classes: Number of classes, including background.
            ignore_index: Target label to exclude from all metrics (e.g. 255).
                Pixels with labels outside ``[0, num_classes)`` are always ignored.
            background_index: Class excluded from ``miou_fg``. ``None`` disables it.
            class_names: Optional human-readable names, used in per-class outputs.
            device: Device on which to keep the confusion matrix.
        """
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but num_classes={num_classes}"
            )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.class_names = list(class_names) if class_names is not None else [
            f"class_{i}" for i in range(num_classes)
        ]
        self.device = torch.device(device) if device is not None else None
        self.confusion_matrix: Optional[torch.Tensor] = None
        self.reset()

    def reset(self) -> None:
        """Zero the confusion matrix."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            preds: Either logits of shape ``(B, C, H, W)`` or predicted labels of
                shape ``(B, H, W)``.
            target: Ground-truth labels of shape ``(B, H, W)``.
        """
        if preds.dim() == target.dim() + 1:
            if preds.shape[1] != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} channels in logits, got {preds.shape[1]}"
                )
            preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {tuple(preds.shape)} vs target {tuple(target.shape)}")

        if self.confusion_matrix.device != target.device:
            if self.device is None:
                # Follow the data the first time we see it (keeps accumulation on GPU).
                self.confusion_matrix = self.confusion_matrix.to(target.device)
            else:
                preds, target = preds.to(self.device), target.to(self.device)

        preds = preds.reshape(-1).long()
        target = target.reshape(-1).long()
        valid = (target >= 0) & (target < self.num_classes)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        preds, target = preds[valid], target[valid]

        indices = target * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes).to(
            self.confusion_matrix.device
        )

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dictionary of python floats / lists (JSON serialisable).
        """
        cm = self.confusion_matrix.double().cpu()
        tp = cm.diag()
        gt_pixels = cm.sum(dim=1)     # TP + FN
        pred_pixels = cm.sum(dim=0)   # TP + FP
        total = cm.sum()
        union = gt_pixels + pred_pixels - tp

        nan = torch.tensor(float("nan"), dtype=torch.float64)
        iou = torch.where(union > 0, tp / union.clamp(min=1), nan)
        dice = torch.where(gt_pixels + pred_pixels > 0, 2 * tp / (gt_pixels + pred_pixels).clamp(min=1), nan)
        recall = torch.where(gt_pixels > 0, tp / gt_pixels.clamp(min=1), nan)
        precision = torch.where(pred_pixels > 0, tp / pred_pixels.clamp(min=1), nan)

        freq = gt_pixels / total if total > 0 else torch.zeros_like(gt_pixels)
        valid_iou = ~torch.isnan(iou)
        fwiou = (freq[valid_iou] * iou[valid_iou]).sum().item() if total > 0 else float("nan")

        fg_mask = torch.ones(self.num_classes, dtype=torch.bool)
        if self.background_index is not None:
            fg_mask[self.background_index] = False

        def _nanmean(x: torch.Tensor) -> float:
            return torch.nanmean(x).item() if (~torch.isnan(x)).any() else float("nan")

        return {
            "pixel_acc": (tp.sum() / total).item() if total > 0 else float("nan"),
            "mean_acc": _nanmean(recall),
            "miou": _nanmean(iou),
            "miou_fg": _nanmean(iou[fg_mask]),
            "fwiou": fwiou,
            "mdice": _nanmean(dice),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_acc": recall.tolist(),
            "per_class_precision": precision.tolist(),
            "class_pixel_freq": freq.tolist(),
            "class_names": list(self.class_names),
            "num_pixels": int(total.item()),
            "confusion_matrix": self.confusion_matrix.cpu().tolist(),
        }


SCALAR_METRICS = ("pixel_acc", "mean_acc", "miou", "miou_fg", "fwiou", "mdice")


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device | str,
    criterion: Optional[nn.Module] = None,
    ignore_index: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run a full evaluation pass and return loss (optional) plus all metrics.

    The loss is averaged per pixel-weighted batch (``sum(loss * B) / N``) so that
    a smaller final batch does not bias it.
    """
    was_training = model.training
    model.eval()
    metric = SegmentationMetrics(
        num_classes=num_classes, ignore_index=ignore_index, class_names=class_names, device=device
    )
    loss_sum, n_samples = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        if criterion is not None:
            loss_sum += criterion(logits, masks).item() * images.shape[0]
            n_samples += images.shape[0]
        metric.update(logits, masks)
    results = metric.compute()
    if criterion is not None:
        results["loss"] = loss_sum / max(n_samples, 1)
    model.train(was_training)
    return results


def format_metrics_table(results: Dict[str, Any], digits: int = 4) -> str:
    """Render summary + per-class metrics as a plain-text table."""
    lines = ["Summary"]
    if "loss" in results:
        lines.append(f"  {'loss':<10} {results['loss']:.{digits}f}")
    for key in SCALAR_METRICS:
        lines.append(f"  {key:<10} {results[key]:.{digits}f}")
    names = results.get("class_names") or [f"class_{i}" for i in range(len(results["per_class_iou"]))]
    width = max(12, max(len(n) for n in names) + 2)
    header = f"\n{'id':>3} {'class':<{width}} {'IoU':>8} {'Dice':>8} {'Recall':>8} {'Prec.':>8} {'GT %':>8}"
    lines += [header, "-" * (len(header) - 1)]
    for i, name in enumerate(names):
        row = [results[k][i] for k in ("per_class_iou", "per_class_dice", "per_class_acc", "per_class_precision")]
        cells = " ".join(f"{v:>8.{digits}f}" if v == v else f"{'n/a':>8}" for v in row)
        lines.append(f"{i:>3} {name:<{width}} {cells} {100 * results['class_pixel_freq'][i]:>7.2f}%")
    return "\n".join(lines)


def get_class_names(dataset: Any, num_classes: int) -> List[str]:
    """Best-effort class names from a synthdataseg dataset's ``ClassRegistry``."""
    registry = getattr(dataset, "registry", None)
    names = []
    for i in range(num_classes):
        try:
            names.append(str(registry.get_class_name(i)))
        except Exception:
            names.append(f"class_{i}")
    return names
