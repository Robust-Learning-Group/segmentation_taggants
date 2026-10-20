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

## Experiment Tracking (Weights & Biases or local)

`logging.backend` in `config.yaml` selects where results go:

- `wandb` (default here): everything below is logged to W&B **and** written to `results/` as usual.
- `local`: only `results/` is written. `wandb` is never imported, so it does not need to be installed.

```bash
wandb login                                                      # once per machine
python experiments/001_pilot_synthetic/run.py                    # logs to W&B
python experiments/001_pilot_synthetic/run.py logging.backend=local
python experiments/001_pilot_synthetic/run.py logging.wandb.mode=offline   # no network; `wandb sync results/001_pilot_synthetic/wandb/offline-run-*` later
python experiments/001_pilot_synthetic/run.py logging.wandb.tags='[lr-sweep]' training.lr=5e-4
```

What is logged (same keys in every experiment, so runs can be compared directly in one W&B project):

| Where | Content |
| :--- | :--- |
| Config | Full resolved Hydra config (`data.*`, `model.*`, `training.*`, `seed`, ...) plus `model/n_params`: filter, group and build parallel-coordinate plots by any hyperparameter. |
| Charts (x = `epoch`) | `train/*` and `val/*` for loss, `pixel_acc`, `mean_acc`, `miou`, `miou_fg`, `fwiou`, `mdice`; `val_iou/<class>` per-class IoU; `lr`. |
| Media | `val/predictions`: fixed validation samples with toggleable ground-truth / prediction mask overlays (every `log_predictions_every` epochs). |
| Final evaluation | `final/per_class` table + IoU bar chart, interactive `final/confusion_matrix` (pixel counts) and row-normalised heat-map, `final/samples` table with per-sample scores (sort to find failures). |
| Run summary | `final/<metric>`, `final/iou/<class>`, `best_epoch`, `best/val_miou_fg`, max/min of `val/miou_fg`, `val/miou`, `val/loss`: these are the columns of the runs table used for cross-experiment comparison. |
| Files / artifacts | `metrics.json`; with `logging.wandb.log_model=true`, final and best checkpoints as a `model` artifact. |

Runs are grouped by `logging.wandb.group` (defaults to `experiment_name`), so seeds and sweeps of one experiment
aggregate together, while different experiments sit side by side in the same project (`logging.wandb.project`).
Set `logging.wandb.entity` if you want to log to a team instead of your personal account.

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

Every run gets its own folder `results/001_pilot_synthetic/<YYYYmmdd-HHMMSS>_<config-hash>/`. The 6-character
hash is computed from the resolved hyperparameters (excluding `device`, `output_dir`, `logging` and `evaluation`),
so runs with identical settings share a suffix and different settings never collide. Sweeps (`--multirun`) go to
`results/001_pilot_synthetic/multirun_<timestamp>/<job>_<config-hash>/`.

```text
results/001_pilot_synthetic/
├── runs.csv                          # one row per finished run: hyperparameters + final metrics (compare runs)
└── 20261020-164232_7f364c/
    ├── .hydra/                       # Hydra config snapshot + overrides
    ├── run.log
    ├── metrics.json                  # full metrics, per-class results, confusion matrix, history
    ├── history.csv                   # per-epoch scalars
    ├── checkpoints/
    │   ├── unet_surrogate.pth        # last epoch
    │   └── unet_surrogate_best.pth   # best validation miou_fg
    ├── plots/
    │   ├── training_curves.png       # loss, mIoU, mIoU(fg), pixel acc, mDice, mean acc (train vs val)
    │   ├── per_class_iou_history.png # val IoU per class over epochs
    │   ├── per_class_metrics.png     # final IoU / Dice / recall / precision per class
    │   ├── confusion_matrix.png      # row-normalised
    │   └── confusion_matrix_counts.png
    ├── predictions/
    │   ├── final.png                 # image | GT | prediction | error map on fixed val samples
    │   └── epoch_XXX.png             # if evaluation.plot_predictions_every > 0
    └── wandb/                        # only with logging.backend=wandb
```

Plot settings live in the `evaluation` section of `config.yaml` (`save_plots`, `num_visual_samples`,
`plot_predictions_every`). With W&B enabled, the same figures are also uploaded under `plots/*` and the W&B run
name defaults to the local folder name, so local and online results are easy to match.

### 3. Fast 1-Epoch Smoke Test (CLI Overrides)

Using Hydra syntax, any hyperparameter can be overridden directly from the CLI (add `logging.backend=local` to keep smoke tests out of W&B):

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
