"""Hydra CLI runner for Experiment 001: Pilot Synthetic Segmentation Training."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plots import (
    plot_per_class_iou_history,
    plot_per_class_metrics,
    plot_prediction_grid,
    plot_training_curves,
)
from src.utils.seed import set_seed
from src.utils.visualize import plot_confusion_matrix
from src.utils.tracking import build_tracker
from src.vision.segmentation.config import (
    ExperimentConfig,
    config_hash,
    register_configs,
    validate_experiment_config,
)
from src.vision.segmentation.metrics import (
    SCALAR_METRICS,
    SegmentationMetrics,
    evaluate_segmentation,
    format_metrics_table,
    get_class_names,
)
from src.vision.segmentation.models import UNetSurrogate

# Register structured configuration schemas with Hydra ConfigStore
register_configs()


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        # prioritizemps if available, otherwise use cuda
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf floats (e.g. IoU of absent classes) with None for strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def resolve_output_dir(cfg: DictConfig, cfg_hash: str) -> Path:
    """Explicit ``output_dir`` > Hydra run dir > results/<experiment>/<timestamp>_<hash> (no Hydra, e.g. notebooks)."""
    if cfg.output_dir:
        return Path(cfg.output_dir)
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:  # HydraConfig not set
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return REPO_ROOT / "results" / str(cfg.experiment_name) / f"{stamp}_{cfg_hash}"


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


def save_result_plots(history, final_val, class_names, palette, plots_dir: Path) -> List[Path]:
    """Write the standard result figures; returns the saved paths."""
    figures = {
        "training_curves.png": lambda p: plot_training_curves(history, save_path=p),
        "per_class_iou_history.png": lambda p: plot_per_class_iou_history(history, class_names, palette, save_path=p),
        "per_class_metrics.png": lambda p: plot_per_class_metrics(final_val, class_names, save_path=p),
        "confusion_matrix.png": lambda p: plot_confusion_matrix(
            final_val["confusion_matrix"], class_names=class_names, normalize=True, save_path=p, show=False),
        "confusion_matrix_counts.png": lambda p: plot_confusion_matrix(
            final_val["confusion_matrix"], class_names=class_names, normalize=False, save_path=p, show=False),
    }
    saved = []
    for name, make in figures.items():
        path = plots_dir / name
        try:
            plt.close(make(path))
            saved.append(path)
        except Exception as err:  # a broken plot must never lose a finished run
            print(f"[plots] could not create {name}: {err}")
    return saved


def write_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    rows = [{k: v for k, v in h.items() if not isinstance(v, list)} for h in history]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def append_runs_index(index_path: Path, row: Dict[str, Any]) -> None:
    """Append one line per run to results/<experiment>/runs.csv for quick local comparison."""
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
        writer.writerow(_json_safe(row))


def train_surrogate(cfg: DictConfig) -> Dict[str, Any]:
    """Train UNetSurrogate model according to Hydra configuration."""
    validate_experiment_config(cfg)
    cfg_hash = config_hash(cfg)
    output_dir = resolve_output_dir(cfg, cfg_hash)
    run_id = output_dir.name
    ckpt_dir, plots_dir, preds_dir = output_dir / "checkpoints", output_dir / "plots", output_dir / "predictions"
    for d in (output_dir, ckpt_dir, plots_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id} (config hash {cfg_hash}) -> {output_dir}")

    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"Training surrogate model on: {device}")
    print("Resolved configuration:\n" + OmegaConf.to_yaml(cfg))

    # Build DataLoaders
    from src.vision.segmentation.data import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        batch_size=cfg.data.batch_size,
        image_size=tuple(cfg.data.image_size),
        num_classes=cfg.data.num_classes,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        min_shapes=cfg.data.min_shapes,
        max_shapes=cfg.data.max_shapes,
        max_n=cfg.data.max_n,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.seed,
    )

    # Initialize UNetSurrogate Model
    model = UNetSurrogate(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        up_channels=cfg.model.up_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    epochs = int(cfg.training.epochs)
    num_classes = int(cfg.model.num_classes)
    class_names = get_class_names(train_loader.dataset, num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    registry = getattr(train_loader.dataset, "registry", None)
    palette = registry.get_color_palette() if hasattr(registry, "get_color_palette") else None

    # Fixed validation samples for qualitative results (deterministic dataset -> same images every run).
    n_vis = min(int(cfg.evaluation.num_visual_samples), len(val_loader.dataset))
    vis_images = vis_masks = None
    if n_vis > 0:
        samples = [val_loader.dataset[i] for i in range(n_vis)]
        vis_images = torch.stack([img for img, _ in samples])
        vis_masks = torch.stack([msk for _, msk in samples])

    tracker = build_tracker(cfg, output_dir, class_names, vis_images, vis_masks, device, run_name=run_id)
    tracker.set_config({
        "model/n_params": n_params,
        "data/class_names": class_names,
        "run/id": run_id,
        "run/config_hash": cfg_hash,
        "run/output_dir": str(output_dir),
    })
    print(f"Experiment tracking backend: {tracker.backend}")
    try:
        history = []
        print(f"Commencing surrogate training for {epochs} epoch(s)...")

        checkpoint_path = ckpt_dir / "unet_surrogate.pth"
        best_checkpoint_path = ckpt_dir / "unet_surrogate_best.pth"
        best = {"epoch": None, "val_miou_fg": float("-inf")}
        train_metric = SegmentationMetrics(num_classes=num_classes, class_names=class_names, device=device)

        for epoch in range(epochs):
            model.train()
            train_metric.reset()
            train_loss_sum, train_samples = 0.0, 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * images.shape[0]
                train_samples += images.shape[0]
                # Running training metrics (model in train mode, weights changing): cheap
                # indicator only -- the validation pass below is the reference evaluation.
                train_metric.update(outputs.detach(), masks)

            train_results = train_metric.compute()
            train_results["loss"] = train_loss_sum / max(train_samples, 1)

            val_results = evaluate_segmentation(
                model, val_loader, num_classes=num_classes, device=device,
                criterion=criterion, class_names=class_names,
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_results['loss']:.4f} | Train mIoU: {train_results['miou']:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | Val PixAcc: {val_results['pixel_acc']:.4f} | "
                f"Val mIoU: {val_results['miou']:.4f} | Val mIoU(fg): {val_results['miou_fg']:.4f} | "
                f"Val mDice: {val_results['mdice']:.4f}"
            )

            record = {"epoch": epoch + 1, "train_loss": train_results["loss"], "val_loss": val_results["loss"]}
            for key in SCALAR_METRICS:
                record[f"train_{key}"] = train_results[key]
                record[f"val_{key}"] = val_results[key]
            record["val_per_class_iou"] = val_results["per_class_iou"]
            history.append(record)
            tracker.log_epoch(epoch + 1, train_results, val_results, lr=optimizer.param_groups[0]["lr"])
            tracker.log_predictions(model, epoch + 1)
            every = int(cfg.evaluation.plot_predictions_every)
            if every > 0 and (epoch + 1) % every == 0:
                save_prediction_grid(model, vis_images, vis_masks, device, class_names, palette,
                                     preds_dir / f"epoch_{epoch + 1:03d}.png",
                                     title=f"{run_id} | validation samples | epoch {epoch + 1}")

            if val_results["miou_fg"] > best["val_miou_fg"]:
                best = {"epoch": epoch + 1, "val_miou_fg": val_results["miou_fg"]}
                torch.save(model.state_dict(), best_checkpoint_path)

        # Save final model checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        if best["epoch"] is not None:
            print(f"Best checkpoint (val mIoU fg={best['val_miou_fg']:.4f} @ epoch {best['epoch']}) "
                  f"saved to {best_checkpoint_path}")

        # Final full evaluation of the last checkpoint on the validation split
        final_val = evaluate_segmentation(
            model, val_loader, num_classes=num_classes, device=device,
            criterion=criterion, class_names=class_names,
        )
        print("Final validation metrics (last epoch):\n" + format_metrics_table(final_val))

        # Save metrics
        metrics = {
            "experiment": str(cfg.experiment_name),
            "run_id": run_id,
            "config_hash": cfg_hash,
            "output_dir": str(output_dir),
            "epochs": epochs,
            "batch_size": int(cfg.data.batch_size),
            "device": str(device),
            "n_params": n_params,
            "logging_backend": str(cfg.logging.backend),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": final_val["loss"],
            "final_val": {k: final_val[k] for k in SCALAR_METRICS},
            "final_val_per_class": {
                name: {
                    "iou": final_val["per_class_iou"][i],
                    "dice": final_val["per_class_dice"][i],
                    "recall": final_val["per_class_acc"][i],
                    "precision": final_val["per_class_precision"][i],
                    "gt_pixel_freq": final_val["class_pixel_freq"][i],
                }
                for i, name in enumerate(class_names)
            },
            "final_val_confusion_matrix": final_val["confusion_matrix"],
            "class_names": class_names,
            "best_epoch": best["epoch"],
            "best_val_miou_fg": best["val_miou_fg"] if best["epoch"] is not None else None,
            "history": history,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(metrics), f, indent=2, allow_nan=False)
        print(f"Metrics saved to {metrics_path}")

        write_history_csv(history, output_dir / "history.csv")

        saved_figures: List[Path] = []
        if cfg.evaluation.save_plots:
            saved_figures = save_result_plots(history, final_val, class_names, palette, plots_dir)
        final_grid = preds_dir / "final.png"
        save_prediction_grid(model, vis_images, vis_masks, device, class_names, palette, final_grid,
                             title=f"{run_id} | validation samples | final (epoch {epochs})")
        if final_grid.exists():
            saved_figures.append(final_grid)
        print(f"Plots saved to {plots_dir} and {preds_dir}")

        experiment_dir = output_dir.parent
        if experiment_dir.name.startswith("multirun_"):  # sweeps share the experiment-level index
            experiment_dir = experiment_dir.parent
        append_runs_index(experiment_dir / "runs.csv", {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "seed": int(cfg.seed),
            "epochs": epochs,
            "batch_size": int(cfg.data.batch_size),
            "lr": float(cfg.training.lr),
            "weight_decay": float(cfg.training.weight_decay),
            "up_channels": int(cfg.model.up_channels),
            "num_classes": num_classes,
            "image_size": "x".join(str(v) for v in cfg.data.image_size),
            "train_samples": int(cfg.data.train_samples),
            "final_val_loss": final_val["loss"],
            **{f"final_val_{k}": final_val[k] for k in SCALAR_METRICS},
            "best_epoch": best["epoch"],
            "best_val_miou_fg": best["val_miou_fg"] if best["epoch"] is not None else None,
            "n_params": n_params,
        })

        tracker.log_predictions(model, epochs, final=True)
        tracker.log_final(final_val, best, output_dir)
        tracker.log_images({f"plots/{p.stem}": p for p in saved_figures})

        print(
            f"Final Summary: Train Loss: {history[-1]['train_loss'] if history else float('nan'):.4f} | "
            f"Val Loss: {final_val['loss']:.4f} | Val PixAcc: {final_val['pixel_acc']:.4f} | "
            f"Val mIoU: {final_val['miou']:.4f} | Val mIoU(fg): {final_val['miou_fg']:.4f} | "
            f"Val mDice: {final_val['mdice']:.4f}"
        )
        tracker.finish()
        return metrics
    except BaseException:
        tracker.finish(exit_code=1)
        raise


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main Hydra entrypoint."""
    schema = OmegaConf.structured(ExperimentConfig)
    validated_cfg = OmegaConf.merge(schema, cfg)
    train_surrogate(validated_cfg)


if __name__ == "__main__":
    main()
