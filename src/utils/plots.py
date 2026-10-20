"""Static result plots for segmentation experiments (saved as PNG into a run folder).

Conventions: train = blue dashed, val = orange solid (identity never colour-only);
class colours come from the dataset palette so charts match the mask overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
FALLBACK_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

PathLike = Union[str, Path]


def _style(ax: plt.Axes, title: str, ylabel: Optional[str] = None, xlabel: Optional[str] = None) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_2, labelsize=9)


def _save(fig: plt.Figure, save_path: Optional[PathLike]) -> plt.Figure:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def class_colors(palette: Optional[np.ndarray], num_classes: int) -> List[str]:
    """Hex colours per class: dataset palette (uint8 RGB) if given, else a fixed categorical order."""
    if palette is not None and len(palette) >= num_classes:
        return ["#%02x%02x%02x" % tuple(int(c) for c in palette[i][:3]) for i in range(num_classes)]
    return [FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)] for i in range(num_classes)]


def colorize(mask: np.ndarray, colors: Sequence[str]) -> np.ndarray:
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in colors])
    return rgb[np.clip(mask, 0, len(colors) - 1)]


def plot_training_curves(history: List[Dict[str, Any]], save_path: Optional[PathLike] = None) -> plt.Figure:
    """Small multiples (one y-scale each): loss, mIoU, foreground mIoU, pixel accuracy, mean Dice, mean accuracy."""
    epochs = [h["epoch"] for h in history]
    panels = [
        ("loss", "Cross-entropy loss"),
        ("miou", "mIoU"),
        ("miou_fg", "mIoU (foreground)"),
        ("pixel_acc", "Pixel accuracy"),
        ("mdice", "Mean Dice"),
        ("mean_acc", "Mean class accuracy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    marker = "o" if len(epochs) <= 30 else None
    for ax, (key, title) in zip(axes.flat, panels):
        for split, color, ls in (("train", TRAIN_COLOR, "--"), ("val", VAL_COLOR, "-")):
            values = [np.nan if h.get(f"{split}_{key}") is None else h[f"{split}_{key}"] for h in history]
            ax.plot(epochs, values, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=5, label=split)
        if key != "loss":
            ax.set_ylim(0, 1)
        _style(ax, title, xlabel="epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Training curves (train metrics are running, train-mode; val is the reference)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_per_class_iou_history(
    history: List[Dict[str, Any]],
    class_names: Sequence[str],
    palette: Optional[np.ndarray] = None,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Validation IoU per class across epochs (one line per class, coloured as in the masks)."""
    epochs = [h["epoch"] for h in history]
    colors = class_colors(palette, len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    marker = "o" if len(epochs) <= 30 else None
    for i, name in enumerate(class_names):
        vals = [h["val_per_class_iou"][i] for h in history]
        vals = [np.nan if v is None else v for v in vals]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(epochs, vals, color=colors[i], linewidth=2, marker=marker, markersize=5, label=f"{i} {name}")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax, "Validation IoU per class", ylabel="IoU", xlabel="epoch")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_per_class_metrics(
    results: Dict[str, Any],
    class_names: Sequence[str],
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Final per-class IoU / Dice / recall / precision as small-multiple horizontal bars."""
    metrics = [("per_class_iou", "IoU"), ("per_class_dice", "Dice"),
               ("per_class_acc", "Recall"), ("per_class_precision", "Precision")]
    n = len(class_names)
    fig, axes = plt.subplots(1, 4, figsize=(14, 0.45 * n + 1.6), sharey=True)
    y = np.arange(n)
    for ax, (key, title) in zip(axes, metrics):
        vals = np.array([np.nan if v is None else v for v in results[key]], dtype=float)
        ax.barh(y, np.nan_to_num(vals), height=0.6, color=TRAIN_COLOR if key == "per_class_iou" else "#86b6ef")
        for yi, v in zip(y, vals):
            ax.text((0 if np.isnan(v) else v) + 0.02, yi, "n/a" if np.isnan(v) else f"{v:.2f}",
                    va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, 1.15)
        ax.set_yticks(y, labels=[f"{i} {c}  ({100 * results['class_pixel_freq'][i]:.1f}% px)"
                                 for i, c in enumerate(class_names)])
        _style(ax, title)
    axes[0].invert_yaxis()  # shared y-axis: invert once so class 0 is on top
    fig.suptitle(
        f"Final validation  |  mIoU {results['miou']:.3f}  ·  mIoU(fg) {results['miou_fg']:.3f}  ·  "
        f"pixel acc {results['pixel_acc']:.3f}  ·  mDice {results['mdice']:.3f}",
        x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_prediction_grid(
    images: np.ndarray,
    gts: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    palette: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Qualitative results: image | ground truth | prediction | errors, one row per sample.

    Args:
        images: (N, H, W, 3) float in [0, 1].
        gts, preds: (N, H, W) integer label maps.
    """
    from src.vision.segmentation.metrics import SegmentationMetrics
    import torch

    n = len(images)
    colors = class_colors(palette, len(class_names))
    fig, axes = plt.subplots(n, 4, figsize=(11, 3.0 * n + 0.8), squeeze=False)
    for i in range(n):
        m = SegmentationMetrics(num_classes=len(class_names))
        m.update(torch.from_numpy(preds[i:i + 1]), torch.from_numpy(gts[i:i + 1]))
        r = m.compute()
        wrong = preds[i] != gts[i]
        # Neutral error map (no hue, so it cannot be confused with a class colour).
        err = np.full(gts[i].shape + (3,), 0.25)
        err[wrong] = 1.0
        panels = [
            (images[i], "image"),
            (colorize(gts[i], colors), "ground truth"),
            (colorize(preds[i], colors), f"prediction  (mIoU {r['miou']:.2f}, fg {r['miou_fg']:.2f})"),
            (err, f"errors: white = wrong  ({100 * wrong.mean():.1f}% px)"),
        ]
        for ax, (img, label) in zip(axes[i], panels):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(f"#{i} {label}" if label == "image" else label, fontsize=9, color=INK_2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():  # thin frame keeps an all-white error map visible
                spine.set_color("#b5b4ae")
                spine.set_linewidth(0.8)
    present = sorted(set(np.unique(gts).tolist()) | set(np.unique(preds).tolist()))
    handles = [Patch(facecolor=colors[c], label=f"{c} {class_names[c]}") for c in present if c < len(class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), h_pad=1.5)
    return _save(fig, save_path)
