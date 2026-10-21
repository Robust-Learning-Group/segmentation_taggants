"""Visualization utilities for segmentation models and watermark inference."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    num_images: int = 3,
    device: Optional[Union[torch.device, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    num_classes: Optional[int] = None,
) -> plt.Figure:
    """Visualize sample inputs, ground-truth masks, and model predictions from a DataLoader.

    Args:
        model: Trained segmentation model.
        loader: DataLoader providing (images, masks).
        num_images: Number of samples to visualize (default: 3).
        device: Device to run inference on (defaults to model's parameter device or CPU).
        save_path: Optional file path to save the generated figure.
        show: Whether to display the plot with plt.show() (default: True).
        num_classes: Number of classes, fixing the colour range to [0, num_classes - 1] for both
            masks so colours are comparable. Defaults to the number of output channels of the model.

    Returns:
        The matplotlib Figure object.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
    if num_classes is None:
        num_classes = logits.shape[1]
    vmax = num_classes - 1

    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    num_images = min(num_images, len(images_np))
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        img_plot = np.transpose(images_np[i], (1, 2, 0))

        axes[i, 0].imshow(img_plot)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks_np[i], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds_np[i], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        axes[i, 2].set_title("U-Net Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def visualize_key_image_inference(
    model: nn.Module,
    image: Union[str, Path, Image.Image],
    num_classes: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    device: Optional[Union[torch.device, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Perform inference on a single key image and visualize side-by-side with color legend.

    Args:
        model: Trained segmentation model.
        image: Path to image file or PIL Image object.
        num_classes: Number of segmentation classes (default: 8).
        image_size: Resize dimensions (H, W) for inference (default: (256, 256)).
        device: Device to run inference on (defaults to model's parameter device or CPU).
        save_path: Optional file path to save the generated figure.
        show: Whether to display the plot with plt.show() (default: True).

    Returns:
        The matplotlib Figure object.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    img_visualize = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_visualize)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(pred, cmap="viridis", vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[1].set_title("Model Segmentation Prediction")
    axes[1].axis("off")

    class_names = [f"Class {i}" for i in range(num_classes)]
    colors = [im.cmap(im.norm(i)) for i in range(num_classes)]
    patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markersize=10, label=class_names[i])
        for i in range(num_classes)
    ]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    confusion_matrix: Union[np.ndarray, torch.Tensor, Sequence[Sequence[int]]],
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a segmentation confusion matrix (rows = ground truth, cols = prediction).

    Args:
        confusion_matrix: (C, C) pixel counts, e.g. ``SegmentationMetrics.compute()["confusion_matrix"]``.
        class_names: Optional class labels for the axes.
        normalize: Row-normalise so each cell is the fraction of GT pixels (per-class recall on the diagonal).
        save_path: Optional file path to save the figure.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure object.
    """
    cm = np.asarray(confusion_matrix.cpu() if isinstance(confusion_matrix, torch.Tensor) else confusion_matrix,
                    dtype=np.float64)
    num_classes = cm.shape[0]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    labels = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * num_classes, 0.8 + 0.7 * num_classes))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else " (pixels)"))
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    if save_path:
        from src.utils.plots import save_figure_atomic

        save_figure_atomic(fig, save_path)
    if show:
        plt.show()
    return fig
