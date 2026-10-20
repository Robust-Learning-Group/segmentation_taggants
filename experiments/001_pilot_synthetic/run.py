"""Hydra CLI runner for Experiment 001: Pilot Synthetic Segmentation Training."""

import json
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
from src.vision.segmentation.models import UNetSurrogate

# Register structured configuration schemas with Hydra ConfigStore
register_configs()


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


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
    history = []
    print(f"Commencing surrogate training for {epochs} epoch(s)...")

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        final_train_loss = avg_train_loss
        final_val_loss = avg_val_loss

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

    # Save model checkpoint
    checkpoint_path = output_dir / "unet_surrogate.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Save metrics
    metrics = {
        "experiment": str(cfg.experiment_name),
        "epochs": epochs,
        "batch_size": int(cfg.data.batch_size),
        "device": str(device),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "history": history,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(f"Final Summary: Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_loss:.4f}")
    return metrics


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main Hydra entrypoint."""
    schema = OmegaConf.structured(ExperimentConfig)
    validated_cfg = OmegaConf.merge(schema, cfg)
    train_surrogate(validated_cfg)


if __name__ == "__main__":
    main()
