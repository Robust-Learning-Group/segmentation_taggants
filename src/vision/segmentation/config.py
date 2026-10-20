"""Structured configuration schemas for semantic segmentation experiments using Hydra and OmegaConf."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


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
class EvaluationConfig:
    """Local evaluation artefacts (plots and qualitative prediction grids)."""

    save_plots: bool = True  # training curves, per-class metrics, confusion matrices -> plots/
    num_visual_samples: int = 8  # fixed validation samples in predictions/*.png (and W&B overlays)
    plot_predictions_every: int = 0  # also save predictions/epoch_XXX.png every N epochs (0 = final only)

    def __post_init__(self) -> None:
        if self.num_visual_samples < 0:
            raise ValueError(f"num_visual_samples must be >= 0, got {self.num_visual_samples}")
        if self.plot_predictions_every < 0:
            raise ValueError(f"plot_predictions_every must be >= 0, got {self.plot_predictions_every}")


LOGGING_BACKENDS = ("local", "wandb")
WANDB_MODES = ("online", "offline", "disabled")


@dataclass
class WandbConfig:
    """Weights & Biases settings (only used when ``logging.backend == "wandb"``)."""

    project: str = "segmentation-taggants"
    entity: Optional[str] = None  # W&B user or team; None -> your default entity
    group: Optional[str] = None  # None -> experiment_name (groups seeds/sweeps of one experiment)
    name: Optional[str] = None  # None -> auto-generated run name
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = "online"  # "online", "offline" (sync later with `wandb sync`), "disabled"
    log_predictions_every: int = 1  # log val prediction overlays every N epochs (0 = final only)
    log_model: bool = False  # upload final + best checkpoints as a W&B model artifact

    def __post_init__(self) -> None:
        validate_wandb_config(self)


@dataclass
class LoggingConfig:
    """Experiment tracking backend. ``local`` writes only to ``results/``; ``wandb`` also logs to W&B."""

    backend: str = "local"  # "local" or "wandb"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        validate_logging_config(self)


def validate_wandb_config(cfg) -> None:
    """Validate W&B settings (works on dataclasses and OmegaConf nodes)."""
    if cfg.mode not in WANDB_MODES:
        raise ValueError(f"logging.wandb.mode must be one of {WANDB_MODES}, got {cfg.mode!r}")
    if cfg.log_predictions_every < 0:
        raise ValueError(f"log_predictions_every must be >= 0, got {cfg.log_predictions_every}")


def validate_logging_config(cfg) -> None:
    """Validate logging settings (works on dataclasses and OmegaConf nodes)."""
    if cfg.backend not in LOGGING_BACKENDS:
        raise ValueError(f"logging.backend must be one of {LOGGING_BACKENDS}, got {cfg.backend!r}")
    validate_wandb_config(cfg.wandb)


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Keys that do not change the trained model / results and are therefore excluded from the config hash.
HASH_EXCLUDED_KEYS = ("hydra", "output_dir", "device", "logging", "evaluation")


def config_hash(cfg: Any, length: int = 6) -> str:
    """Short, stable hash of the hyperparameters (identical settings -> identical hash)."""
    container = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    payload = {k: v for k, v in container.items() if k not in HASH_EXCLUDED_KEYS}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:length]


def _config_hash_resolver(*, _root_: Any) -> str:
    # _root_ includes the `hydra` node, which must not be resolved here (it references this resolver).
    payload = {k: _root_[k] for k in _root_ if k not in HASH_EXCLUDED_KEYS}
    return config_hash(OmegaConf.create(OmegaConf.to_container(OmegaConf.create(payload), resolve=True)))


def validate_experiment_config(cfg: Any) -> None:
    """Cross-field checks that the per-dataclass validators cannot see."""
    if cfg.model.num_classes != cfg.data.num_classes:
        raise ValueError(
            f"model.num_classes ({cfg.model.num_classes}) must equal data.num_classes ({cfg.data.num_classes})"
        )
    if cfg.data.max_n is not None and cfg.data.max_n != cfg.data.num_classes - 1:
        raise ValueError(
            f"data.max_n ({cfg.data.max_n}) must be data.num_classes - 1 ({cfg.data.num_classes - 1}): "
            "classes are background + star + crown + N-gons with 3..max_n vertices"
        )


def register_configs() -> None:
    """Register structured dataclass schemas with Hydra's ConfigStore (and the ``config_hash`` resolver)."""
    if not OmegaConf.has_resolver("config_hash"):
        OmegaConf.register_new_resolver("config_hash", _config_hash_resolver, use_cache=False)
    cs = ConfigStore.instance()
    cs.store(name="base_experiment_config", node=ExperimentConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="evaluation", name="base_evaluation", node=EvaluationConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
