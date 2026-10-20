# Experiment 001: Pilot Synthetic Segmentation Training

## Overview

This experiment tests the training of a compact U-Net surrogate model (`UNetSurrogate`) on a procedural synthetic segmentation dataset generated on-the-fly by `synthdataseg`.

Hyperparameters are managed and validated via **Hydra** (`hydra-core`) with structured dataclass schemas defined in `src/vision/segmentation/config.py`.

## Dataset and Configuration

- **Dataset:** Procedural multi-shape synthetic semantic segmentation via `synthdataseg.dataset.SyntheticSegmentationDataset`.
- **Image Size:** 256x256 RGB (`data.image_size=[256, 256]`).
- **Classes:** 8 classes (background + 7 shape classes, `data.num_classes=8`).
- **Dataset Size:** 8,000 training images, 1,600 validation images.
- **Shapes per image:** 1 to 4 shapes (`data.min_shapes=1`, `data.max_shapes=4`).
- **Architecture:** `UNetSurrogate` (`model.up_channels=32`, `model.in_channels=3`).
- **Optimization:** AdamW (`training.lr=1e-3`, `training.weight_decay=1e-4`), `nn.CrossEntropyLoss`.
- **Default Batch Size:** 64 (`data.batch_size=64`).
- **Default Epochs:** 10 (`training.epochs=10`).

## Baseline Results (Observed in Original Notebook Run)

In the exploratory Colab notebook (`SemanticSegmentation_Synth.ipynb`), the observed training progression and loss metrics were:

| Epoch | Train Loss (Cross-Entropy) | Val Loss (Cross-Entropy) | Notes |
| :---: | :---: | :---: | :--- |
| **1 / 10** | 1.1542 | 0.7212 | First training pass |
| **5 / 10** | 0.4787 | 0.4780 | |
| **10 / 10** | **0.3873** | **0.3906** | **Baseline 10-epoch checkpoint** |
| **20 / 20** | **0.3426** | **0.3404** | Observed after 10 additional fine-tuning epochs |

*(Note: These figures are labeled as **observed in the original notebook run** on Google Colab GPU).*

## How to Reproduce with Hydra

### 1. Inspect Configuration Without Running

```bash
python experiments/001_pilot_synthetic/run.py --cfg job
```

### 2. Full 10-Epoch Training Run (Default Config)

```bash
python experiments/001_pilot_synthetic/run.py
```

Outputs will be saved directly to `results/001_pilot_synthetic/`:
- Model weights: `results/001_pilot_synthetic/unet_surrogate.pth`
- Metrics & resolved configuration: `results/001_pilot_synthetic/metrics.json`
- Hydra run logs & configuration snapshot: `results/001_pilot_synthetic/.hydra/`

### 3. Fast 1-Epoch Smoke Test (CLI Overrides)

Using Hydra syntax, any hyperparameter can be overridden directly from the CLI:

```bash
python experiments/001_pilot_synthetic/run.py training.epochs=1
```

### 4. Custom Parameter Examples

```bash
# Run on CPU with smaller batch size and custom learning rate:
python experiments/001_pilot_synthetic/run.py device=cpu data.batch_size=32 training.lr=0.0005

# Hydra multirun hyperparameter sweep:
python experiments/001_pilot_synthetic/run.py --multirun training.lr=1e-3,5e-4
```
