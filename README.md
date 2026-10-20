# Data Taggants for Semantic Segmentation

This repository adapts the **Data Taggants** dataset watermarking framework to **semantic segmentation**. It builds cleanly upon the original *Data Taggants: Concealing and Detecting Watermarks in Machine Learning Datasets* method while keeping the original classification-oriented paper implementation as an untouched, read-only baseline.

## Overview

In classification data taggants, clean dataset poisoning introduces a subtle trigger to key samples causing a watermark detector to verify model provenance via hypothesis testing. In semantic segmentation, we adapt this framework to pixel-level dense prediction tasks using synthetic segmentation datasets and differentiable U-Net surrogate models.

Key components:
- **Baseline (`baseline/`):** The original classification paper implementation preserved verbatim as a read-only reference.
- **Library (`src/`):** Modular components for domain-agnostic watermark theory (`src/core/`), segmentation surrogates and loaders (`src/vision/segmentation/`), and visualization/reproducibility tools (`src/utils/`).
- **Experiments (`experiments/`):** Self-contained, reproducible experiment configs, execution scripts, and metric trackers.

---

## Directory Structure

```text
segmentation_taggants/
├── baseline/                     # Original paper code (read-only baseline, do not edit)
│   ├── README.md                 # Baseline reference notice
│   ├── README_original.md        # Original paper documentation
│   ├── requirements.txt          # Original paper dependencies
│   └── src/                      # Original classification modules
├── src/
│   ├── __init__.py
│   ├── core/                     # Domain-agnostic math & statistics
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract interfaces for generators & detectors
│   │   ├── detection.py          # Binomial tests, p-value calculations
│   │   └── keys.py               # Key generation logic & schemas
│   ├── vision/                   # Vision-specific adapters
│   │   ├── __init__.py
│   │   └── segmentation/         # Segmentation extension
│   │       ├── __init__.py
│   │       ├── config.py         # Structured Hydra & OmegaConf schemas
│   │       ├── data.py           # Dataloader wrapper around synthdataseg
│   │       ├── models.py         # DoubleConv and UNetSurrogate
│       ├── metrics.py        # Confusion-matrix metrics (mIoU, Dice, pixel acc.)
│   │       └── taggants.py       # Segmentation taggant generator logic
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── seed.py               # Deterministic seed configuration
│       ├── tracking.py           # Local / Weights & Biases experiment trackers
│       ├── plots.py              # Training curves, per-class metrics, prediction grids
│       └── visualize.py          # Prediction & key inference visualization
├── experiments/
│   └── 001_pilot_synthetic/      # Pilot surrogate training on synthetic shapes
│       ├── README.md             # Experiment overview & baseline benchmarks
│       ├── config.yaml           # Hyperparameters & data configuration
│       └── run.py                # Hydra CLI runner for training & evaluation
├── tests/
│   └── test_segmentation_metrics.py  # Unit tests for segmentation metrics
├── notebooks/
│   ├── 01_smoke_test.ipynb               # 1-epoch pipeline + metrics smoke test
│   ├── SemanticSegmentation_Synth.ipynb  # Original exploratory Colab notebook
│   └── reference/
│       └── synthetic_training.ipynb      # Reference notebook copy
├── data/                         # Local dataset storage (git-ignored)
├── results/                      # Checkpoints and metrics outputs (git-ignored)
├── requirements.txt              # Pinned project dependencies
├── requirements-wandb.txt        # Optional W&B tracking dependency
└── README.md                     # Project documentation (this file)
```

---

## Installation

Dependencies are declared in `pyproject.toml` and locked in `uv.lock` (managed with [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/hallpaz/segmentation_taggants.git
cd segmentation_taggants

# Creates .venv and installs everything, including the default groups:
#   dev      -> pytest, ipykernel
#   tracking -> wandb (only needed for logging.backend=wandb)
uv sync
wandb login            # once per machine, only if you log to W&B

# Environment without Weights & Biases (use logging.backend=local)
uv sync --no-group tracking
```

Run commands with `uv run`, e.g. `uv run python experiments/001_pilot_synthetic/run.py` or `uv run pytest`.
`requirements.txt` / `requirements-wandb.txt` are kept for plain `pip install -r` setups.

---

## Running the Pilot Experiment

Experiments use **Hydra** (`hydra-core`) to manage and validate hyperparameters:

```bash
# Inspect resolved configuration without running
python experiments/001_pilot_synthetic/run.py --cfg job

# Full 10-epoch training run (matches notebook baseline)
python experiments/001_pilot_synthetic/run.py

# Fast 1-epoch smoke test with CLI override
python experiments/001_pilot_synthetic/run.py training.epochs=1

# Custom device or hyperparameter overrides
python experiments/001_pilot_synthetic/run.py device=cpu data.batch_size=32 training.lr=0.0005

# Local results only (wandb is not imported, need not be installed)
python experiments/001_pilot_synthetic/run.py logging.backend=local
```

Experiment tracking is selected with `logging.backend` in each experiment's `config.yaml`: `wandb` logs to Weights & Biases
(project `segmentation-taggants`, one W&B group per experiment), `local` writes only to `results/`. See the experiment README for what is logged.

Each run writes checkpoints, metrics, plots and qualitative predictions to its own folder `results/001_pilot_synthetic/<timestamp>_<config-hash>/`, and appends a summary row to `results/001_pilot_synthetic/runs.csv`.

---

## Reference Baseline

The original paper code is housed in [`baseline/`](baseline/). It is preserved strictly as a read-only reference implementation. Do not edit files inside `baseline/`.
