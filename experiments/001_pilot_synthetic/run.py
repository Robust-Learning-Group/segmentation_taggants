"""CLI runner for Experiment 001: Pilot Synthetic Segmentation Training."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import set_seed
from src.vision.segmentation.data import build_dataloaders
from src.vision.segmentation.models import UNetSurrogate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 001_pilot_synthetic surrogate segmentation training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "001_pilot_synthetic"),
        help="Directory to save checkpoints and metrics.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train_surrogate(
    config: Dict[str, Any],
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    device_override: str | None = None,
    output_dir: str | Path = REPO_ROOT / "results" / "001_pilot_synthetic",
) -> Dict[str, Any]:
    """Train UNetSurrogate model according to configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Determine device
    device_name = device_override or config.get("device", "auto")
    device = resolve_device(device_name)
    print(f"Training surrogate model on: {device}")

    # Data parameters
    data_cfg = config.get("data", {})
    batch_size = batch_size_override or data_cfg.get("batch_size", 64)
    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    num_classes = data_cfg.get("num_classes", 8)
    max_n = data_cfg.get("max_n", num_classes - 1)
    min_shapes = data_cfg.get("min_shapes", 1)
    max_shapes = data_cfg.get("max_shapes", 4)
    train_samples = data_cfg.get("train_samples", 8000)
    val_samples = data_cfg.get("val_samples", 1600)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)

    train_loader, val_loader = build_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        train_samples=train_samples,
        val_samples=val_samples,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        max_n=max_n,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    # Model parameters
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    up_channels = model_cfg.get("up_channels", 32)
    model = UNetSurrogate(
        in_channels=in_channels,
        num_classes=num_classes,
        up_channels=up_channels,
    ).to(device)

    # Training parameters
    train_cfg = config.get("training", {})
    epochs = epochs_override if epochs_override is not None else train_cfg.get("epochs", 10)
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    checkpoint_path = output_path / "unet_surrogate.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Save metrics
    metrics = {
        "experiment": config.get("experiment_name", "001_pilot_synthetic"),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "history": history,
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print(
        f"Final Summary: Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_loss:.4f}"
    )
    return metrics


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    train_surrogate(
        config=config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        device_override=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
