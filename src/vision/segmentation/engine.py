"""Training engine shared by the segmentation experiments (``experiments/*/run.py`` are thin Hydra wrappers).

Run folder layout (``results/<experiment>/<YYYYmmdd-HHMMSS>_<config-hash>/``)::

    checkpoints/last.pth              full training state (model, optimizer, scheduler, AMP scaler, RNG states,
                                      history, best, W&B run id) -- written every `training.checkpoint_every` epochs
    checkpoints/unet_surrogate_best.pth   weights with the best validation mIoU(fg)
    checkpoints/unet_surrogate.pth    final weights
    history.csv, progress.json        updated every epoch (cheap; for monitoring a running job)
    plots/                            refreshed every `evaluation.plot_every` epochs and at the end
    predictions/epoch_XXX.png         every `evaluation.plot_predictions_every` epochs; final.png at the end
    metrics.json                      written last -> marks the run as finished

Resuming (`training.resume`): ``auto`` continues the newest unfinished run with the same config hash (if any),
or pass a run folder / checkpoint path. Training continues from the next epoch with the same history, best
checkpoint, LR schedule, data order and augmentation stream, and the same W&B run.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.utils.plots import (
    plot_per_class_iou_history,
    plot_per_class_metrics,
    plot_prediction_grid,
    plot_training_curves,
)
from src.utils.seed import set_seed
from src.utils.tracking import build_tracker
from src.utils.visualize import plot_confusion_matrix
from src.vision.segmentation.config import config_hash, resolve_run_dir, validate_experiment_config
from src.vision.segmentation.data import build_datasets, make_loader, to_model_input
from src.vision.segmentation.losses import SegmentationLoss, random_flip_rot90
from src.vision.segmentation.metrics import (
    SCALAR_METRICS,
    SegmentationMetrics,
    evaluate_segmentation,
    format_metrics_table,
    get_class_names,
)
from src.vision.segmentation.models import UNetSurrogate

SELECT_METRIC = "miou_fg"  # model selection (best checkpoint)


# --------------------------------------------------------------------------- small helpers
def resolve_device(device_str: str) -> torch.device:
    """'auto' prefers MPS (Apple GPU), then CUDA, then CPU."""
    if device_str == "auto":
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf floats (e.g. IoU of absent classes) with None for strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write to a temp file then rename, so a crash/disconnect never leaves a corrupt checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def resolve_output_dir(cfg: DictConfig) -> Path:
    """Explicit ``output_dir`` > Hydra run dir > same rule as Hydra (resume-aware) when run without Hydra."""
    if cfg.output_dir:
        return Path(cfg.output_dir)
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except (ImportError, ValueError):  # HydraConfig not set (e.g. called from a notebook)
        return resolve_run_dir(cfg, datetime.now().strftime("%Y%m%d-%H%M%S"))


def build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig, steps_per_epoch: int):
    """Per-step LR schedule: constant, or linear warmup followed by cosine decay to lr * min_lr_ratio."""
    total = max(1, int(cfg.training.epochs) * steps_per_epoch)
    warmup = int(float(cfg.training.warmup_epochs) * steps_per_epoch)
    min_ratio = float(cfg.training.min_lr_ratio)

    def factor(step: int) -> float:
        if cfg.training.scheduler == "none":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def rng_state(loader_gen: torch.Generator, aug_gen: torch.Generator) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": loader_gen.get_state(),
        "augment": aug_gen.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any], loader_gen: torch.Generator, aug_gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_gen.set_state(state["loader"])
    aug_gen.set_state(state["augment"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError:
            pass  # different number of GPUs than when saved


@torch.no_grad()
def predict_labels(model: nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    was_training = model.training
    model.eval()
    preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
    model.train(was_training)
    return preds


def save_prediction_grid(model, vis_images, vis_masks, device, class_names, palette, path: Path, title: str) -> None:
    if vis_images is None or len(vis_images) == 0:
        return
    preds = predict_labels(model, vis_images, device)
    images_np = vis_images.permute(0, 2, 3, 1).clamp(0, 1).numpy()
    fig = plot_prediction_grid(images_np, vis_masks.numpy(), preds, class_names, palette, title=title, save_path=path)
    plt.close(fig)


def save_result_plots(history, results, class_names, palette, plots_dir: Path) -> List[Path]:
    """Write the standard result figures (overwriting previous versions); returns the saved paths."""
    figures = {
        "training_curves.png": lambda p: plot_training_curves(history, save_path=p),
        "per_class_iou_history.png": lambda p: plot_per_class_iou_history(history, class_names, palette, save_path=p),
        "per_class_metrics.png": lambda p: plot_per_class_metrics(results, class_names, save_path=p),
        "confusion_matrix.png": lambda p: plot_confusion_matrix(
            results["confusion_matrix"], class_names=class_names, normalize=True, save_path=p, show=False),
        "confusion_matrix_counts.png": lambda p: plot_confusion_matrix(
            results["confusion_matrix"], class_names=class_names, normalize=False, save_path=p, show=False),
    }
    saved = []
    for name, make in figures.items():
        path = plots_dir / name
        try:
            plt.close(make(path))
            saved.append(path)
        except Exception as err:  # a broken plot must never lose a run
            print(f"[plots] could not create {name}: {err}")
    return saved


def write_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    rows = [{k: v for k, v in h.items() if not isinstance(v, list)} for h in history]
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    os.replace(tmp, path)


def append_runs_index(index_path: Path, row: Dict[str, Any]) -> None:
    """Append one line per finished run to results/<experiment>/runs.csv for quick local comparison."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    exists = index_path.exists()
    if exists:
        with open(index_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            # Columns changed (new metric/hyperparameter): start a new index instead of corrupting the old one.
            index_path = index_path.with_name(f"runs_{datetime.now():%Y%m%d-%H%M%S}.csv")
            exists = False
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(json_safe(row))


def _per_class(results: Dict[str, Any], class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "iou": results["per_class_iou"][i],
            "dice": results["per_class_dice"][i],
            "recall": results["per_class_acc"][i],
            "precision": results["per_class_precision"][i],
            "gt_pixel_freq": results["class_pixel_freq"][i],
        }
        for i, name in enumerate(class_names)
    }


# --------------------------------------------------------------------------- main entry point
def train_segmentation(cfg: DictConfig) -> Dict[str, Any]:
    """Train (or resume) a UNetSurrogate according to ``cfg`` and write all results to the run folder."""
    validate_experiment_config(cfg)
    cfg_hash = config_hash(cfg)
    output_dir = resolve_output_dir(cfg)
    run_id = output_dir.name
    ckpt_dir, plots_dir, preds_dir = output_dir / "checkpoints", output_dir / "plots", output_dir / "predictions"
    for d in (output_dir, ckpt_dir, plots_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pth"
    best_ckpt_path = ckpt_dir / "unet_surrogate_best.pth"
    final_ckpt_path = ckpt_dir / "unet_surrogate.pth"

    # ---- which checkpoint (if any) to resume from
    resume = cfg.training.resume
    resume_file: Optional[Path] = None
    if resume:
        explicit = Path(str(resume)).expanduser()
        if resume != "auto" and explicit.is_file():
            resume_file = explicit
        elif last_ckpt_path.exists():
            resume_file = last_ckpt_path
        elif resume != "auto":
            raise FileNotFoundError(f"training.resume={resume!r} but no checkpoint found in {output_dir}")

    print(f"Run: {run_id} (config hash {cfg_hash}) -> {output_dir}")
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")
    print("Resolved configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # ---- data
    t0 = time.time()
    datasets = build_datasets(
        image_size=tuple(cfg.data.image_size), num_classes=int(cfg.data.num_classes),
        train_samples=int(cfg.data.train_samples), val_samples=int(cfg.data.val_samples),
        test_samples=int(cfg.data.test_samples), min_shapes=int(cfg.data.min_shapes),
        max_shapes=int(cfg.data.max_shapes), max_n=cfg.data.max_n, seed=int(cfg.seed),
        cache_dir=cfg.data.cache_dir, generation_workers=int(cfg.data.generation_workers),
    )
    print(f"Datasets ready in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))
    loader_gen = torch.Generator().manual_seed(int(cfg.seed))
    aug_gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    pin = bool(cfg.data.pin_memory) or device.type == "cuda"
    train_loader = make_loader(datasets["train"], int(cfg.data.batch_size), True, int(cfg.data.num_workers), pin, loader_gen)
    val_loader = make_loader(datasets["val"], int(cfg.data.batch_size), False, int(cfg.data.num_workers), pin)
    test_loader = (make_loader(datasets["test"], int(cfg.data.batch_size), False, int(cfg.data.num_workers), pin)
                   if "test" in datasets else None)

    num_classes = int(cfg.model.num_classes)
    class_names = get_class_names(datasets["train"], num_classes)
    registry = getattr(datasets["train"], "registry", None)
    palette = registry.get_color_palette() if hasattr(registry, "get_color_palette") else None

    # ---- model / optimisation
    model = UNetSurrogate(in_channels=int(cfg.model.in_channels), num_classes=num_classes,
                          up_channels=int(cfg.model.up_channels), depth=int(cfg.model.depth)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = SegmentationLoss(num_classes, kind=str(cfg.training.loss), dice_weight=float(cfg.training.dice_weight))
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Model: UNetSurrogate(up_channels={cfg.model.up_channels}, depth={cfg.model.depth}) "
          f"{n_params / 1e6:.2f}M params | loss={cfg.training.loss} | scheduler={cfg.training.scheduler}")

    # ---- state (fresh or resumed)
    epochs = int(cfg.training.epochs)
    start_epoch = 0
    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": None, f"val_{SELECT_METRIC}": float("-inf")}
    wandb_run_id: Optional[str] = None
    if resume_file is not None:
        state = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and use_amp:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        history = list(state["history"])
        best = dict(state["best"])
        wandb_run_id = state.get("wandb_run_id")
        restore_rng_state(state["rng"], loader_gen, aug_gen)
        if state.get("config_hash") != cfg_hash:
            print(f"WARNING: resuming a run trained with config hash {state.get('config_hash')} using {cfg_hash}; "
                  "changed hyperparameters (e.g. epochs) apply from now on.")
        print(f"Resumed from {resume_file} at epoch {start_epoch}/{epochs} "
              f"(best val {SELECT_METRIC}={best[f'val_{SELECT_METRIC}']:.4f} @ epoch {best['epoch']})")
    elif resume == "auto":
        print("training.resume=auto: no unfinished run with this config found -> starting a new run")

    # Fixed validation samples for qualitative results (deterministic dataset -> same images every run).
    n_vis = min(int(cfg.evaluation.num_visual_samples), len(datasets["val"]))
    vis_images = vis_masks = None
    if n_vis > 0:
        samples = [datasets["val"][i] for i in range(n_vis)]
        vis_images, vis_masks = to_model_input(torch.stack([s[0] for s in samples]),
                                               torch.stack([s[1] for s in samples]), torch.device("cpu"))

    tracker = build_tracker(cfg, output_dir, class_names, vis_images, vis_masks, device,
                            run_name=run_id, resume_id=wandb_run_id)
    wandb_run_id = tracker.run_id or wandb_run_id
    if wandb_run_id:
        atomic_write_text(output_dir / "wandb_run_id.txt", wandb_run_id + "\n")
    tracker.set_config({"model/n_params": n_params, "data/class_names": class_names, "run/id": run_id,
                        "run/config_hash": cfg_hash, "run/output_dir": str(output_dir)})
    print(f"Experiment tracking backend: {tracker.backend}")

    def save_last(epoch_done: int, completed: bool = False) -> None:
        atomic_torch_save({
            "epoch": epoch_done, "completed": completed,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if use_amp else None,
            "history": history, "best": best, "rng": rng_state(loader_gen, aug_gen),
            "wandb_run_id": wandb_run_id, "config_hash": cfg_hash,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, last_ckpt_path)

    try:
        train_metric = SegmentationMetrics(num_classes=num_classes, class_names=class_names, device=device)
        if start_epoch < epochs:
            print(f"Training epochs {start_epoch + 1}..{epochs} ({len(train_loader)} steps/epoch)")
        epoch_times = [h["epoch_time_s"] for h in history if h.get("epoch_time_s")]

        for epoch in range(start_epoch, epochs):
            t_epoch = time.time()
            model.train()
            train_metric.reset()
            loss_sum, n_seen = 0.0, 0
            for images, masks in train_loader:
                images, masks = to_model_input(images, masks, device)
                if cfg.data.augment:
                    images, masks = random_flip_rot90(images, masks, generator=aug_gen)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item() * images.shape[0]
                n_seen += images.shape[0]
                # Running train-mode metrics: a cheap progress indicator; validation is the reference.
                train_metric.update(outputs.detach(), masks)

            train_results = train_metric.compute()
            train_results["loss"] = loss_sum / max(n_seen, 1)
            val_results = evaluate_segmentation(model, val_loader, num_classes=num_classes, device=device,
                                                criterion=criterion, class_names=class_names)
            epoch_time = time.time() - t_epoch
            epoch_times.append(epoch_time)
            lr_now = optimizer.param_groups[0]["lr"]

            record = {"epoch": epoch + 1, "lr": lr_now, "epoch_time_s": epoch_time,
                      "train_loss": train_results["loss"], "val_loss": val_results["loss"]}
            for key in SCALAR_METRICS:
                record[f"train_{key}"] = train_results[key]
                record[f"val_{key}"] = val_results[key]
            record["val_per_class_iou"] = val_results["per_class_iou"]
            history.append(record)

            improved = val_results[SELECT_METRIC] > best[f"val_{SELECT_METRIC}"]
            if improved:
                best = {"epoch": epoch + 1, f"val_{SELECT_METRIC}": val_results[SELECT_METRIC],
                        "val_miou": val_results["miou"]}
                atomic_torch_save(model.state_dict(), best_ckpt_path)

            eta_min = (epochs - epoch - 1) * (sum(epoch_times[-5:]) / len(epoch_times[-5:])) / 60
            print(f"Epoch {epoch + 1}/{epochs} | {epoch_time:.0f}s | lr {lr_now:.2e} | "
                  f"train loss {train_results['loss']:.4f} mIoU {train_results['miou']:.4f} | "
                  f"val loss {val_results['loss']:.4f} mIoU {val_results['miou']:.4f} "
                  f"mIoU(fg) {val_results['miou_fg']:.4f} pixAcc {val_results['pixel_acc']:.4f}"
                  f"{' *best*' if improved else ''} | ETA {eta_min:.0f} min", flush=True)

            # cheap per-epoch artefacts for monitoring
            write_history_csv(history, output_dir / "history.csv")
            atomic_write_text(output_dir / "progress.json", json.dumps(json_safe({
                "run_id": run_id, "epoch": epoch + 1, "epochs": epochs, "eta_min": eta_min,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "val": {k: val_results[k] for k in SCALAR_METRICS}, "best": best,
                "val_per_class_iou": dict(zip(class_names, val_results["per_class_iou"])),
            }), indent=2))

            tracker.log_epoch(epoch + 1, train_results, val_results, lr=lr_now)
            tracker.log_predictions(model, epoch + 1)
            every_pred = int(cfg.evaluation.plot_predictions_every)
            if every_pred > 0 and (epoch + 1) % every_pred == 0:
                save_prediction_grid(model, vis_images, vis_masks, device, class_names, palette,
                                     preds_dir / f"epoch_{epoch + 1:03d}.png",
                                     title=f"{run_id} | validation samples | epoch {epoch + 1}")
            every_plot = int(cfg.evaluation.plot_every)
            if cfg.evaluation.save_plots and every_plot > 0 and (epoch + 1) % every_plot == 0:
                save_result_plots(history, val_results, class_names, palette, plots_dir)

            if (epoch + 1) % int(cfg.training.checkpoint_every) == 0 or epoch + 1 == epochs:
                save_last(epoch + 1)

        # ---- final evaluation
        atomic_torch_save(model.state_dict(), final_ckpt_path)
        save_last(max(epochs, start_epoch), completed=True)
        final_val = evaluate_segmentation(model, val_loader, num_classes=num_classes, device=device,
                                          criterion=criterion, class_names=class_names)
        print("Final validation metrics (last epoch):\n" + format_metrics_table(final_val))

        test_best = None
        if test_loader is not None:
            best_model = UNetSurrogate(in_channels=int(cfg.model.in_channels), num_classes=num_classes,
                                       up_channels=int(cfg.model.up_channels), depth=int(cfg.model.depth)).to(device)
            best_model.load_state_dict(torch.load(best_ckpt_path if best_ckpt_path.exists() else final_ckpt_path,
                                                  map_location=device, weights_only=True))
            test_best = evaluate_segmentation(best_model, test_loader, num_classes=num_classes, device=device,
                                              criterion=criterion, class_names=class_names)
            print(f"Test metrics (best checkpoint, epoch {best['epoch']}):\n" + format_metrics_table(test_best))
            del best_model

        write_history_csv(history, output_dir / "history.csv")
        saved_figures: List[Path] = []
        if cfg.evaluation.save_plots:
            saved_figures = save_result_plots(history, final_val, class_names, palette, plots_dir)
        final_grid = preds_dir / "final.png"
        save_prediction_grid(model, vis_images, vis_masks, device, class_names, palette, final_grid,
                             title=f"{run_id} | validation samples | final (epoch {max(epochs, start_epoch)})")
        if final_grid.exists():
            saved_figures.append(final_grid)

        metrics = {
            "experiment": str(cfg.experiment_name), "run_id": run_id, "config_hash": cfg_hash,
            "output_dir": str(output_dir), "epochs": epochs, "batch_size": int(cfg.data.batch_size),
            "device": str(device), "n_params": n_params, "logging_backend": str(cfg.logging.backend),
            "wandb_run_id": wandb_run_id, "resumed": resume_file is not None,
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": final_val["loss"],
            "final_val": {k: final_val[k] for k in SCALAR_METRICS},
            "final_val_per_class": _per_class(final_val, class_names),
            "final_val_confusion_matrix": final_val["confusion_matrix"],
            "test_best": ({k: test_best[k] for k in SCALAR_METRICS + ("loss",)} if test_best else None),
            "test_best_per_class": _per_class(test_best, class_names) if test_best else None,
            "test_best_confusion_matrix": test_best["confusion_matrix"] if test_best else None,
            "class_names": class_names,
            "best_epoch": best["epoch"],
            f"best_val_{SELECT_METRIC}": best[f"val_{SELECT_METRIC}"] if best["epoch"] is not None else None,
            "history": history,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        experiment_dir = output_dir.parent
        if experiment_dir.name.startswith("multirun_"):  # sweeps share the experiment-level index
            experiment_dir = experiment_dir.parent
        append_runs_index(experiment_dir / "runs.csv", {
            "run_id": run_id, "config_hash": cfg_hash,
            "finished_at": datetime.now().isoformat(timespec="seconds"), "device": str(device),
            "seed": int(cfg.seed), "epochs": epochs, "batch_size": int(cfg.data.batch_size),
            "lr": float(cfg.training.lr), "weight_decay": float(cfg.training.weight_decay),
            "loss": str(cfg.training.loss), "scheduler": str(cfg.training.scheduler),
            "augment": bool(cfg.data.augment), "up_channels": int(cfg.model.up_channels),
            "depth": int(cfg.model.depth), "num_classes": num_classes,
            "image_size": "x".join(str(v) for v in cfg.data.image_size),
            "shapes": f"{cfg.data.min_shapes}-{cfg.data.max_shapes}",
            "train_samples": int(cfg.data.train_samples),
            "final_val_loss": final_val["loss"],
            **{f"final_val_{k}": final_val[k] for k in SCALAR_METRICS},
            "best_epoch": best["epoch"],
            f"best_val_{SELECT_METRIC}": best[f"val_{SELECT_METRIC}"] if best["epoch"] is not None else None,
            **{f"test_best_{k}": (test_best[k] if test_best else None) for k in SCALAR_METRICS},
            "n_params": n_params,
        })

        tracker.log_predictions(model, max(epochs, start_epoch), final=True)
        tracker.log_final(final_val, best, output_dir, test=test_best)
        tracker.log_images({f"plots/{p.stem}": p for p in saved_figures})

        # metrics.json is written last: its presence marks the run as finished (auto-resume skips it).
        with open(output_dir / "metrics.json.tmp", "w", encoding="utf-8") as f:
            json.dump(json_safe(metrics), f, indent=2, allow_nan=False)
        os.replace(output_dir / "metrics.json.tmp", output_dir / "metrics.json")

        summary = (f"Final Summary: val mIoU {final_val['miou']:.4f} | val mIoU(fg) {final_val['miou_fg']:.4f} | "
                   f"val pixAcc {final_val['pixel_acc']:.4f}")
        if test_best:
            summary += f" | test (best ckpt) mIoU {test_best['miou']:.4f} mIoU(fg) {test_best['miou_fg']:.4f}"
        print(summary)
        print(f"Results in {output_dir}")
        tracker.finish()
        return metrics
    except BaseException:
        tracker.finish(exit_code=1)
        raise
