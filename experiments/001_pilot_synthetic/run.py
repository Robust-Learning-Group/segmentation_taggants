"""Hydra CLI runner for Experiment 001: Pilot Synthetic Segmentation Training."""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import set_seed
from src.vision.segmentation.config import ExperimentConfig, register_configs
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


def train_surrogate(cfg: DictConfig) -> Dict[str, Any]:
    """Train UNetSurrogate model according to Hydra configuration."""
    output_dir = Path(cfg.output_dir) if cfg.output_dir else REPO_ROOT / "results" / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

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
    history = []
    print(f"Commencing surrogate training for {epochs} epoch(s)...")

    checkpoint_path = output_dir / "unet_surrogate.pth"
    best_checkpoint_path = output_dir / "unet_surrogate_best.pth"
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
        "epochs": epochs,
        "batch_size": int(cfg.data.batch_size),
        "device": str(device),
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

    print(
        f"Final Summary: Train Loss: {history[-1]['train_loss'] if history else float('nan'):.4f} | "
        f"Val Loss: {final_val['loss']:.4f} | Val PixAcc: {final_val['pixel_acc']:.4f} | "
        f"Val mIoU: {final_val['miou']:.4f} | Val mIoU(fg): {final_val['miou_fg']:.4f} | "
        f"Val mDice: {final_val['mdice']:.4f}"
    )
    return metrics


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main Hydra entrypoint."""
    schema = OmegaConf.structured(ExperimentConfig)
    validated_cfg = OmegaConf.merge(schema, cfg)
    train_surrogate(validated_cfg)


if __name__ == "__main__":
    main()
