# Experiment 002: 6-class shape segmentation with a compact U-Net

**Goal:** mIoU > 0.85 (ideally > 0.90) on synthetic shapes, with a surrogate small enough (~2M parameters) for
later taggant optimisation. Run it on Colab with `notebooks/02_colab_train_shapes6.ipynb`, or locally:

```bash
uv run python experiments/002_shapes6_unet/run.py                      # resumes automatically if interrupted
uv run python experiments/002_shapes6_unet/run.py logging.backend=wandb
```

## Task

| | |
| :--- | :--- |
| Classes | 0 background, 1 star, 2 crown, 3 triangle, 4 quadrilateral, 5 pentagon (`num_classes=6`, `max_n=5`) |
| Images | 256×256, **2–3 shapes per image**, random procedural texture or plain colour per shape |
| Splits | train 5000 (= 1000 × 5 shape classes), val 1000, test 1000 — disjoint seeds (42, 1042, 2042) |
| Balance | each shape class appears in ~50% of images and covers ~5% of pixels; background ~74% |

`min_shapes=2` guarantees every image has adjacency/occlusion (the generator covers 10–45% of the image with
foreground and paints shapes back to front), which is what forces the model to use shape rather than local cues.
Textures are random per shape, so colour/texture carry no class information.

## Hyperparameters and rationale

| Setting | Value | Why |
| :--- | :--- | :--- |
| Model | `UNetSurrogate(up_channels=16, depth=4)`, **1.94M params** | Same size as the 001 model (1.93M) but one more downsampling level. Distinguishing a pentagon from a quadrilateral requires seeing the whole shape (shapes are up to ~120 px across). Measured receptive field: **92 px** for the 001 architecture (depth 3) vs **188 px** for depth 4; the effective receptive field is smaller still, so depth 3 is the main risk for the target. Going to 8M parameters (width 32) does not enlarge the receptive field and was not needed. |
| Loss | cross-entropy + soft Dice (weight 1) | CE gives stable per-pixel gradients; Dice weighs every class equally regardless of its ~5% pixel share and directly optimises overlap, which is what mIoU measures. |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | Standard, robust setting for a BatchNorm U-Net trained from scratch. |
| Schedule | 3-epoch linear warmup, cosine decay to 1e-5, stepped per batch | Warmup avoids early BatchNorm/Adam instabilities; the cosine tail is where most of the final IoU is gained on boundaries. |
| Epochs / batch | 100 epochs × 313 steps (batch 16) ≈ 31k steps | Small model + plenty of steps; 100 epochs ≈ 45–70 min on a T4 (estimate: ~21 GFLOP per training image). |
| Augmentation | random flips + 90° rotations (D4) | Exactly label-preserving for every class and interpolation-free; effectively 8× more distinct views of the 5000 images. |
| Mixed precision | fp16 autocast + GradScaler (CUDA only) | ~2× faster on T4; losses are computed in fp32. |
| Model selection | best validation **mIoU(fg)** (background excluded) | Background IoU is ~0.97 and inflates mIoU; selecting on the shape classes is stricter. The best checkpoint is evaluated once on the held-out **test** split at the end. |

mIoU is reported **with** background (`miou`, the standard definition used for the goal) and **without**
(`miou_fg`). Everything is computed from a dataset-level confusion matrix (see `src/vision/segmentation/metrics.py`).

## Performance notes

Rendering costs ~65 ms per image on one CPU core, so regenerating 5000 images per epoch would leave the GPU idle on
Colab (2 cores). With `data.cache_dir`, every split is rendered **once** (in parallel), saved as uint8 tensors
(~1.3 GB train, ~0.26 GB each for val/test) and served from memory. Samples are deterministic per index, so the cached
dataset is identical to the on-the-fly one.

## Periodic inspection

| Every | Output |
| :--- | :--- |
| epoch | console line (loss, mIoU, mIoU(fg), pixel acc, ETA), `history.csv`, `progress.json`, `checkpoints/last.pth`; W&B scalars + per-class IoU |
| 5 epochs (`evaluation.plot_every`) | `plots/` refreshed: training curves, per-class IoU over epochs, per-class metrics, confusion matrices |
| 10 epochs (`evaluation.plot_predictions_every`) | `predictions/epoch_XXX.png`: the same 8 validation images (image / GT / prediction / errors); W&B mask overlays |
| end | `predictions/final.png`, final plots, test metrics, `metrics.json`, row in `runs.csv` |

## Checkpoints and resuming

`checkpoints/last.pth` holds the full training state (model, optimizer, LR scheduler, AMP scaler, RNG states for
data order and augmentation, metric history, best-so-far, W&B run id). With `training.resume: auto` (default here),
starting the same configuration again continues the newest **unfinished** run of that configuration in the **same
folder** and the **same W&B run**: epochs, charts and `history.csv` keep counting. A resumed run reproduces an
uninterrupted one exactly (verified on CPU: identical per-epoch metrics and test results).

* Fresh run despite an unfinished one: `training.resume=null`.
* Continue a specific run (e.g. train longer): `training.resume=<run folder or checkpoint> training.epochs=150`
  (the cosine schedule is stretched from the current step; a warning notes the changed config hash).

## Known issue: ambiguous polygon classes

This experiment uses `synthdataseg` at commit `d5ed00f` (v0.1.0), whose N-gon classes are uniformly random convex
polygons (Valtr's algorithm). Only ~57% of quadrilaterals and ~32% of pentagons show all their corners, so the
polygon labels are partly ambiguous and polygon IoU stays low (epoch 61: triangle 0.49, quadrilateral 0.30,
pentagon 0.34). Kept as the baseline (`data.polygon_sampler: "valtr"`, which reproduces the v0.1.0 images exactly with synthdataseg >= 0.2.0); see [`003_shapes6_wellformed`](../003_shapes6_wellformed/README.md).

## Results

| Run | val mIoU | val mIoU(fg) | test mIoU (best ckpt) | test mIoU(fg) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *pending first Colab run* | | | | | |
