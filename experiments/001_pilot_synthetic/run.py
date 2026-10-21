"""Hydra CLI runner for Experiment 001: Pilot Synthetic Segmentation Training.

All training / evaluation / logging logic lives in ``src/vision/segmentation/engine.py``.
"""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.segmentation.config import ExperimentConfig, register_configs  # noqa: E402
from src.vision.segmentation.engine import train_segmentation  # noqa: E402

# Register structured configuration schemas (and the run_dir / config_hash resolvers) before Hydra starts.
register_configs()


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main Hydra entrypoint: validate against the schema, then train."""
    validated_cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), cfg)
    train_segmentation(validated_cfg)


if __name__ == "__main__":
    main()
