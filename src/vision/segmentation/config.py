"""Structured configuration schemas for semantic segmentation experiments using Hydra and OmegaConf."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    test_samples: int = 0  # held-out split (own seed), evaluated once at the end with the best checkpoint
    num_workers: int = 0
    pin_memory: bool = False
    cache_dir: Optional[str] = None  # render each split once and serve it from memory (None = on-the-fly)
    generation_workers: int = 0  # processes used to render the cache (0 = all CPU cores)
    augment: bool = False  # random flips + 90-degree rotations on the training batches (label-preserving)
    # How synthdataseg samples the convex N-gon classes: "wellformed" (N clearly visible corners),
    # "regular" or "valtr" (uniform random, <= v0.1.0 behaviour). None = library default (needs synthdataseg >= 0.2
    # for any explicit value).
    polygon_sampler: Optional[str] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must have 2 dimensions (H, W), got {self.image_size}")
        if self.min_shapes > self.max_shapes:
            raise ValueError(f"min_shapes ({self.min_shapes}) cannot exceed max_shapes ({self.max_shapes})")
        if min(self.train_samples, self.val_samples) <= 0 or self.test_samples < 0:
            raise ValueError("train_samples and val_samples must be > 0 and test_samples >= 0")


@dataclass
class ModelConfig:
    """Configuration for UNetSurrogate segmentation architecture."""

    in_channels: int = 3
    num_classes: int = 8
    up_channels: int = 32
    depth: int = 3  # number of 2x downsamplings (receptive field: 3 -> 92 px, 4 -> 188 px, 5 -> 380 px)

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.up_channels <= 0:
            raise ValueError(f"up_channels must be positive, got {self.up_channels}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")


@dataclass
class TrainingConfig:
    """Configuration for optimizer and training loop."""

    epochs: int = 10
    lr: float = 0.001  # peak learning rate (AdamW)
    weight_decay: float = 0.0001
    loss: str = "ce"  # "ce" or "ce_dice"
    dice_weight: float = 1.0  # weight of the Dice term for loss="ce_dice"
    scheduler: str = "none"  # "none" (constant lr) or "cosine" (linear warmup + cosine decay, per step)
    warmup_epochs: float = 0.0
    min_lr_ratio: float = 0.01  # final lr = lr * min_lr_ratio for the cosine schedule
    amp: bool = False  # mixed precision (fp16 autocast + GradScaler); only used on CUDA
    grad_clip: Optional[float] = None  # max gradient norm (None = no clipping)
    checkpoint_every: int = 1  # save checkpoints/last.pth every N epochs (always at the end)
    # Resume: None = fresh run; "auto" = continue the newest unfinished run with the same config hash, if any;
    # or a path to a run folder / checkpoint file (e.g. to extend training with more epochs).
    resume: Optional[str] = None

    def __post_init__(self) -> None:
        if self.loss not in ("ce", "ce_dice"):
            raise ValueError(f"training.loss must be 'ce' or 'ce_dice', got {self.loss!r}")
        if self.scheduler not in ("none", "cosine"):
            raise ValueError(f"training.scheduler must be 'none' or 'cosine', got {self.scheduler!r}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
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
    # Qualitative predictions every N epochs: predictions/epoch_XXX.png and, with W&B, val/predictions overlays.
    # 0 = final only (predictions/final.png and W&B final/predictions are always produced).
    plot_predictions_every: int = 0
    # Refresh plots/ (curves, per-class metrics, confusion matrices) every N epochs (0 = only at the end).
    plot_every: int = 0

    def __post_init__(self) -> None:
        if self.num_visual_samples < 0:
            raise ValueError(f"num_visual_samples must be >= 0, got {self.num_visual_samples}")
        if self.plot_predictions_every < 0:
            raise ValueError(f"plot_predictions_every must be >= 0, got {self.plot_predictions_every}")
        if self.plot_every < 0:
            raise ValueError(f"plot_every must be >= 0, got {self.plot_every}")


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
    device: str = "auto"  # 'auto' (mps > cuda > cpu), 'mps', 'cuda', 'cpu'
    output_dir: Optional[str] = None
    results_root: str = "results"  # parent of results/<experiment>/<run>; e.g. a Google Drive folder on Colab
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Keys that do not change the trained model / results and are therefore excluded from the config hash
# (where results go, how they are logged/plotted, caching and loader performance knobs, resuming).
HASH_EXCLUDED_KEYS = (
    "hydra", "output_dir", "results_root", "device", "logging", "evaluation",
    "data.cache_dir", "data.generation_workers", "data.num_workers", "data.pin_memory",
    "training.resume", "training.checkpoint_every",
)


def _drop_dotted(container: dict, dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    node = container
    for key in parents:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict):
        node.pop(leaf, None)


def _drop_none(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _drop_none(v) for k, v in node.items() if v is not None}
    return node


def config_hash(cfg: Any, length: int = 6) -> str:
    """Short, stable hash of the hyperparameters (identical settings -> identical hash).

    Unset (None) values are ignored, so adding a new optional setting to the schema does not change the hash of
    existing configurations (which would break auto-resume of their runs).
    """
    container = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else json.loads(json.dumps(cfg))
    for key in HASH_EXCLUDED_KEYS:
        _drop_dotted(container, key)
    container = _drop_none(container)
    digest = hashlib.sha1(json.dumps(container, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:length]


def _job_config(_root_: Any) -> Any:
    """The job config (no ``hydra`` node) merged with the schema, as the runner will see it."""
    payload = OmegaConf.create({k: _root_[k] for k in _root_ if k != "hydra"})
    payload = OmegaConf.create(OmegaConf.to_container(payload, resolve=True))
    return OmegaConf.merge(OmegaConf.structured(ExperimentConfig), payload)


def _config_hash_resolver(*, _root_: Any) -> str:
    # _root_ includes the `hydra` node, which must not be resolved here (it references this resolver).
    return config_hash(_job_config(_root_))


def resume_dir_from_path(path: str | Path) -> Path:
    """Run folder for a resume target: a run folder, its checkpoints/ folder, or a checkpoint file."""
    p = Path(path).expanduser()
    if p.is_file():
        return p.parent.parent if p.parent.name == "checkpoints" else p.parent
    if p.name == "checkpoints":
        return p.parent
    if (p / "checkpoints" / "last.pth").exists():
        return p
    raise FileNotFoundError(f"training.resume={path!r}: no checkpoint file or run folder with checkpoints/last.pth")


def find_resumable_run(results_root: str | Path, experiment_name: str, cfg_hash: str) -> Optional[Path]:
    """Newest run folder with this config hash that has a checkpoint but did not finish (no metrics.json)."""
    exp_dir = Path(results_root) / experiment_name
    if not exp_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in exp_dir.glob(f"*_{cfg_hash}") if d.is_dir() and not d.name.startswith("multirun_")),
        key=lambda d: d.name,
    )
    for run in reversed(candidates):
        if (run / "checkpoints" / "last.pth").exists() and not (run / "metrics.json").exists():
            return run
    return None


def resolve_run_dir(cfg: Any, stamp: str) -> Path:
    """Run folder for a (merged) job config: resumed run if requested/found, else a new timestamped folder."""
    cfg_hash = config_hash(cfg)
    resume = cfg.training.resume
    if resume and resume != "auto":
        return resume_dir_from_path(resume)
    if resume == "auto":
        found = find_resumable_run(cfg.results_root, cfg.experiment_name, cfg_hash)
        if found is not None:
            return found
    return Path(cfg.results_root) / str(cfg.experiment_name) / f"{stamp}_{cfg_hash}"


def _run_dir_resolver(stamp: str, *, _root_: Any) -> str:
    # Used as hydra.run.dir: ${run_dir:${now:%Y%m%d-%H%M%S}}
    return str(resolve_run_dir(_job_config(_root_), stamp))


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
    if not OmegaConf.has_resolver("run_dir"):
        OmegaConf.register_new_resolver("run_dir", _run_dir_resolver, use_cache=False)
    cs = ConfigStore.instance()
    cs.store(name="base_experiment_config", node=ExperimentConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="evaluation", name="base_evaluation", node=EvaluationConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
