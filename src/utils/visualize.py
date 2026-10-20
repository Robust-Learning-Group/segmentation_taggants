"""Visualization utilities for segmentation models and watermark inference."""

from pathlib import Path
from typing import Optional, Tuple, Union
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
) -> plt.Figure:
    """Visualize sample inputs, ground-truth masks, and model predictions from a DataLoader.

    Args:
        model: Trained segmentation model.
        loader: DataLoader providing (images, masks).
        num_images: Number of samples to visualize (default: 3).
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

    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

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

        axes[i, 1].imshow(masks_np[i], cmap="viridis", vmin=0, vmax=3)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds_np[i], cmap="viridis", vmin=0, vmax=3)
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

    im = axes[1].imshow(pred, cmap="viridis", vmin=0, vmax=num_classes - 1)
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
