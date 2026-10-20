"""Semantic segmentation adaptation of data taggants."""

from src.vision.segmentation.models import DoubleConv, UNetSurrogate

__all__ = ["DoubleConv", "UNetSurrogate", "build_dataloaders"]


def __getattr__(name: str):
    if name == "build_dataloaders":
        from src.vision.segmentation.data import build_dataloaders
        return build_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
