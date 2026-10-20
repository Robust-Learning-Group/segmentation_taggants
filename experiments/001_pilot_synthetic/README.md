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

## Evaluation Metrics

Evaluation is implemented in `src/vision/segmentation/metrics.py` (`SegmentationMetrics`, `evaluate_segmentation`).
All metrics derive from a single 8x8 confusion matrix accumulated over the **whole validation split** (rows = ground
truth, columns = prediction), so IoU uses dataset-level TP/FP/FN and does not depend on batch size. The validation
pass runs with the model in `eval()` mode after every epoch.

| Key | Metric | Notes |
| :--- | :--- | :--- |
| `pixel_acc` | Pixel accuracy | Inflated by background (~70% of pixels); an all-background predictor scores ~0.72. |
| `mean_acc` | Mean class accuracy (mean recall) | |
| `miou` | Mean IoU over all classes | Standard headline number. |
| `miou_fg` | Mean IoU excluding background | **Primary metric**, used to select `unet_surrogate_best.pth`. |
| `fwiou` | Frequency-weighted IoU | Weighted by GT pixel share. |
| `mdice` | Mean Dice / F1 | |

Per-class IoU, Dice, recall, precision and GT pixel share are also reported. Classes absent from both ground truth
and predictions get an undefined (NaN) IoU, are excluded from the means, and are stored as `null` in `metrics.json`.
Training-set metrics in `history` are *running* metrics accumulated during the epoch in train mode (cheap progress
indicator, not a clean evaluation).

`metrics.json` contains `final_val` (summary), `final_val_per_class`, `final_val_confusion_matrix`, `class_names`,
`best_epoch` / `best_val_miou_fg`, and a per-epoch `history` with `train_*` / `val_*` versions of every summary metric.
Unit tests against hand-computed values: `python -m pytest tests/test_segmentation_metrics.py`.

## Baseline Results (Observed in Original Notebook Run)

In the exploratory Colab notebook (`SemanticSegmentation_Synth.ipynb`), the observed training progression and loss metrics were:

| Epoch | Train Loss (Cross-Entropy) | Val Loss (Cross-Entropy) | Notes |
| :---: | :---: | :---: | :--- |
| **1 / 10** | 1.1542 | 0.7212 | First training pass |
| **5 / 10** | 0.4787 | 0.4780 | |
| **10 / 10** | **0.3873** | **0.3906** | **Baseline 10-epoch checkpoint** |
| **20 / 20** | **0.3426** | **0.3404** | Observed after 10 additional fine-tuning epochs |

*The original notebook only tracked loss; IoU-based baselines will be filled in from the first full run with the metrics above.*

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
- Model weights (last epoch): `results/001_pilot_synthetic/unet_surrogate.pth`
- Best weights by validation `miou_fg`: `results/001_pilot_synthetic/unet_surrogate_best.pth`
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
