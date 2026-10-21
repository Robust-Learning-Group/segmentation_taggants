"""Abstract interfaces for watermark embedding, extraction, and evaluation."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTaggantGenerator(ABC):
    """Abstract base class for taggant / watermark key generators."""

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate a taggant key or pattern."""
        raise NotImplementedError


class BaseDetector(ABC):
    """Abstract base class for watermark statistical detectors."""

    @abstractmethod
    def detect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform hypothesis testing and return test statistics."""
        raise NotImplementedError
