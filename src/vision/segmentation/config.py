"""Structured configuration schemas for semantic segmentation experiments using Hydra and OmegaConf."""

from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Configuration for dataset generation and PyTorch dataloaders."""

    batch_size: int = 64
    num_classes: int = 8
    image_size: List[int] = field(default_factory=lambda: [256, 256])
    max_n: int = 7
    min_shapes: int = 1
    max_shapes: int = 4
    train_samples: int = 8000
    val_samples: int = 1600
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must have 2 dimensions (H, W), got {self.image_size}")
        if self.min_shapes > self.max_shapes:
            raise ValueError(f"min_shapes ({self.min_shapes}) cannot exceed max_shapes ({self.max_shapes})")


@dataclass
class ModelConfig:
    """Configuration for UNetSurrogate segmentation architecture."""

    in_channels: int = 3
    num_classes: int = 8
    up_channels: int = 32

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.up_channels <= 0:
            raise ValueError(f"up_channels must be positive, got {self.up_channels}")


@dataclass
class TrainingConfig:
    """Configuration for optimizer and training loop."""

    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"learning rate lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")


@dataclass
class ExperimentConfig:
    """Top-level configuration schema for synthetic segmentation surrogate experiments."""

    experiment_name: str = "001_pilot_synthetic"
    seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    output_dir: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs() -> None:
    """Register structured dataclass schemas with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_experiment_config", node=ExperimentConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
