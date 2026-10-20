# Experiment 001: Pilot Synthetic Segmentation Training

## Overview

This experiment tests the training of a compact U-Net surrogate model (`UNetSurrogate`) on a procedural synthetic segmentation dataset generated on-the-fly by `synthdataseg`.

The surrogate model serves as the differentiable segmentation proxy needed for subsequent bilevel optimization and watermark embedding (taggant generation).

## Dataset and Configuration

- **Dataset:** Procedural multi-shape synthetic semantic segmentation via `synthdataseg.dataset.SyntheticSegmentationDataset`.
- **Image Size:** 256x256 RGB.
- **Classes:** 8 classes (background + 7 shape classes).
- **Dataset Size:** 8,000 training images, 1,600 validation images.
- **Shapes per image:** 1 to 4 shapes.
- **Architecture:** `UNetSurrogate` (base width `up_channels=32`).
- **Optimization:** AdamW (`lr=1e-3`, `weight_decay=1e-4`), `nn.CrossEntropyLoss`.
- **Default Batch Size:** 64.
- **Default Epochs:** 10.

## Baseline Results (Observed in Original Notebook Run)

In the exploratory Colab notebook (`SemanticSegmentation_Synth.ipynb`), the observed training progression and loss metrics were:

| Epoch | Train Loss (Cross-Entropy) | Val Loss (Cross-Entropy) | Notes |
| :---: | :---: | :---: | :--- |
| **1 / 10** | 1.1542 | 0.7212 | First training pass |
| **5 / 10** | 0.4787 | 0.4780 | |
| **10 / 10** | **0.3873** | **0.3906** | **Baseline 10-epoch checkpoint** |
| **20 / 20** | **0.3426** | **0.3404** | Observed after 10 additional fine-tuning epochs |

*(Note: These figures are labeled as **observed in the original notebook run** on Google Colab GPU).*

## How to Reproduce

### 1. Full 10-Epoch Training Run

```bash
python experiments/001_pilot_synthetic/run.py --config experiments/001_pilot_synthetic/config.yaml
```

Checkpoints will be written to `results/001_pilot_synthetic/unet_surrogate.pth` and metrics to `results/001_pilot_synthetic/metrics.json`.

### 2. Fast 1-Epoch Smoke Test (CPU or Quick Validation)

```bash
python experiments/001_pilot_synthetic/run.py --config experiments/001_pilot_synthetic/config.yaml --epochs 1
```
