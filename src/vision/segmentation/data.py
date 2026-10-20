"""Dataset and DataLoader wrappers around synthdataseg for synthetic segmentation."""

from typing import Optional, Tuple
from torch.utils.data import DataLoader
try:
    from synthdataseg.dataset import SyntheticSegmentationDataset
except ImportError as err:
    raise ImportError(
        "synthdataseg is required to build synthetic segmentation datasets.\n"
        "Please install the dependencies listed in requirements.txt (including synthdataseg)."
    ) from err


def build_dataloaders(
    batch_size: int = 64,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 8,
    train_samples: int = 8000,
    val_samples: int = 1600,
    min_shapes: int = 1,
    max_shapes: int = 4,
    max_n: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders for synthetic segmentation training and validation.

    Wraps synthdataseg.dataset.SyntheticSegmentationDataset for on-the-fly generation.

    Args:
        batch_size: Number of samples per batch (default: 64).
        image_size: Spatial dimensions (H, W) of synthetic images (default: (256, 256)).
        num_classes: Total number of segmentation classes including background (default: 8).
        train_samples: Virtual length of training dataset (default: 8000).
        val_samples: Virtual length of validation dataset (default: 1600).
        min_shapes: Minimum number of geometric shapes per image (default: 1).
        max_shapes: Maximum number of geometric shapes per image (default: 4).
        max_n: Maximum class index for shapes. Defaults to `num_classes - 1`.
        num_workers: Number of subprocess workers for DataLoader (default: 0).
        pin_memory: Whether to pin memory in DataLoader for faster GPU transfer (default: False).
        seed: Optional base random seed.

    Returns:
        A tuple of `(train_loader, val_loader)`.
    """
    if max_n is None:
        max_n = num_classes - 1

    train_seed = seed if seed is not None else None
    val_seed = (seed + 1000) if seed is not None else None

    train_dataset = SyntheticSegmentationDataset(
        num_samples=train_samples,
        image_size=image_size,
        max_n=max_n,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        seed=train_seed,
    )

    val_dataset = SyntheticSegmentationDataset(
        num_samples=val_samples,
        image_size=image_size,
        max_n=max_n,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        seed=val_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
