"""Semantic segmentation adaptation of data taggants."""

from src.vision.segmentation.models import DoubleConv, UNetSurrogate
from src.vision.segmentation.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    register_configs,
)

__all__ = [
    "DoubleConv",
    "UNetSurrogate",
    "build_dataloaders",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "register_configs",
]


def __getattr__(name: str):
    if name == "build_dataloaders":
        from src.vision.segmentation.data import build_dataloaders
        return build_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
