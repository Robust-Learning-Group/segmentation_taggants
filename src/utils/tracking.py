"""Experiment tracking backends: local-only (default) and Weights & Biases.

``wandb`` is imported lazily inside :class:`WandbTracker`, so with
``logging.backend=local`` it is never imported and does not need to be installed.

W&B layout (identical keys across experiments so runs can be compared side by side):

* ``train/<metric>``, ``val/<metric>``  -- per-epoch scalars (x-axis: ``epoch``)
* ``val_iou/<class>``                    -- per-epoch per-class IoU
* ``val/predictions``                    -- image / GT / prediction mask overlays
* ``final/<metric>`` (run summary)        -- last-epoch validation metrics, used in the runs table
* ``final/per_class``, ``final/confusion_matrix`` -- tables / charts of the final evaluation
* ``run.config``                          -- full resolved Hydra config (+ ``model/n_params``)

Runs are grouped by ``logging.wandb.group`` (default: ``experiment_name``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.vision.segmentation.config import validate_logging_config
from src.vision.segmentation.metrics import SCALAR_METRICS


class LocalTracker:
    """No-op tracker: results are only written locally (``metrics.json``, checkpoints)."""

    backend = "local"

    def log_epoch(self, epoch: int, train: Dict[str, Any], val: Dict[str, Any], lr: Optional[float] = None) -> None:
        pass

    def log_predictions(self, model: nn.Module, epoch: int, final: bool = False) -> None:
        pass

    def log_final(self, final_val: Dict[str, Any], best: Dict[str, Any], output_dir: Path) -> None:
        pass

    def set_config(self, values: Dict[str, Any]) -> None:
        pass

    def log_images(self, images: Dict[str, Path]) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass


class WandbTracker(LocalTracker):
    """Logs metrics, prediction overlays, tables and (optionally) checkpoints to W&B."""

    backend = "wandb"

    def __init__(
        self,
        cfg: DictConfig,
        output_dir: Path,
        class_names: Sequence[str],
        sample_images: Optional[torch.Tensor] = None,
        sample_masks: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        run_name: Optional[str] = None,
    ) -> None:
        try:
            import wandb
        except ImportError as err:
            raise ImportError(
                "logging.backend='wandb' but the `wandb` package is not installed.\n"
                "Install it (e.g. `uv pip install wandb`) and run `wandb login`, "
                "or run with `logging.backend=local`."
            ) from err

        self.wandb = wandb
        self.cfg = cfg
        self.wcfg = cfg.logging.wandb
        self.class_names = list(class_names)
        self.class_labels = {i: name for i, name in enumerate(self.class_names)}
        self.sample_images = sample_images
        self.sample_masks = sample_masks
        self.device = device

        output_dir.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=self.wcfg.project,
            entity=self.wcfg.entity,
            group=self.wcfg.group or str(cfg.experiment_name),
            name=self.wcfg.name or run_name,  # default: local run folder name (timestamp_confighash)
            tags=list(self.wcfg.tags) if self.wcfg.tags else None,
            notes=self.wcfg.notes,
            mode=self.wcfg.mode,
            job_type="train",
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit="finish_previous",  # safe for Hydra --multirun (several runs in one process)
        )
        # Per-epoch x-axis and run-table summaries.
        self.run.define_metric("epoch")
        self.run.define_metric("train/*", step_metric="epoch")
        self.run.define_metric("val/*", step_metric="epoch")
        self.run.define_metric("val_iou/*", step_metric="epoch")
        self.run.define_metric("lr", step_metric="epoch")
        self.run.define_metric("val/miou_fg", step_metric="epoch", summary="max")
        self.run.define_metric("val/miou", step_metric="epoch", summary="max")
        self.run.define_metric("val/loss", step_metric="epoch", summary="min")

    # ------------------------------------------------------------------ helpers
    def set_config(self, values: Dict[str, Any]) -> None:
        self.run.config.update(values, allow_val_change=True)

    def log_images(self, images: Dict[str, Path]) -> None:
        """Upload locally rendered figures (training curves, prediction grids, ...)."""
        payload = {key: self.wandb.Image(str(path)) for key, path in images.items() if Path(path).exists()}
        if payload:
            self.run.log(payload)

    def _overlay(self, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, caption: str):
        return self.wandb.Image(
            image,
            masks={
                "ground_truth": {"mask_data": gt.astype(np.uint8), "class_labels": self.class_labels},
                "prediction": {"mask_data": pred.astype(np.uint8), "class_labels": self.class_labels},
            },
            caption=caption,
        )

    # ------------------------------------------------------------------ logging
    def log_epoch(self, epoch: int, train: Dict[str, Any], val: Dict[str, Any], lr: Optional[float] = None) -> None:
        log: Dict[str, Any] = {"epoch": epoch, "train/loss": train["loss"], "val/loss": val["loss"]}
        for key in SCALAR_METRICS:
            log[f"train/{key}"] = train[key]
            log[f"val/{key}"] = val[key]
        for name, iou in zip(self.class_names, val["per_class_iou"]):
            if iou == iou:  # skip NaN (class absent from GT and predictions)
                log[f"val_iou/{name}"] = iou
        if lr is not None:
            log["lr"] = lr
        self.run.log(log)


    @torch.no_grad()
    def log_predictions(self, model: nn.Module, epoch: int, final: bool = False) -> None:
        if self.sample_images is None or len(self.sample_images) == 0:
            return
        every = int(self.cfg.evaluation.plot_predictions_every)  # shared with the local PNGs
        if not final and (every <= 0 or epoch % every != 0):
            return
        was_training = model.training
        model.eval()
        images = self.sample_images.to(self.device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        model.train(was_training)

        images_np = images.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
        gts = self.sample_masks.cpu().numpy()
        overlays = [
            self._overlay(images_np[i], gts[i], preds[i], caption=f"val sample {i} | epoch {epoch}")
            for i in range(len(images_np))
        ]
        if final:
            # Per-sample table: sortable by score to find failure cases.
            from src.vision.segmentation.metrics import SegmentationMetrics

            table = self.wandb.Table(columns=["sample", "overlay", "pixel_acc", "miou", "miou_fg"])
            for i in range(len(images_np)):
                m = SegmentationMetrics(num_classes=len(self.class_names))
                m.update(torch.from_numpy(preds[i:i + 1]), torch.from_numpy(gts[i:i + 1]))
                r = m.compute()
                table.add_data(i, overlays[i], r["pixel_acc"], r["miou"], r["miou_fg"])
            self.run.log({"final/predictions": overlays, "final/samples": table})
        else:
            self.run.log({"epoch": epoch, "val/predictions": overlays})

    def log_final(self, final_val: Dict[str, Any], best: Dict[str, Any], output_dir: Path) -> None:
        wandb = self.wandb
        for key in SCALAR_METRICS:
            self.run.summary[f"final/{key}"] = final_val[key]
        self.run.summary["final/loss"] = final_val["loss"]
        self.run.summary["best_epoch"] = best.get("epoch")
        if best.get("epoch") is not None:
            self.run.summary["best/val_miou_fg"] = best["val_miou_fg"]

        # Per-class table + bar chart.
        per_class = wandb.Table(columns=["class_id", "class", "iou", "dice", "recall", "precision", "gt_pixel_freq"])
        for i, name in enumerate(self.class_names):
            per_class.add_data(
                i, name,
                *[_nan_to_none(final_val[k][i]) for k in
                  ("per_class_iou", "per_class_dice", "per_class_acc", "per_class_precision")],
                final_val["class_pixel_freq"][i],
            )
            self.run.summary[f"final/iou/{name}"] = final_val["per_class_iou"][i]

        # Confusion matrix as an interactive W&B chart (built from pixel counts, not raw pixels).
        cm = np.asarray(final_val["confusion_matrix"], dtype=np.int64)
        cm_table = wandb.Table(columns=["Actual", "Predicted", "nPredictions"])
        for i, actual in enumerate(self.class_names):
            for j, predicted in enumerate(self.class_names):
                cm_table.add_data(actual, predicted, int(cm[i, j]))

        log = {
            "final/per_class": per_class,
            "final/per_class_iou_chart": wandb.plot.bar(per_class, "class", "iou", title="Final per-class IoU (val)"),
            "final/confusion_matrix": wandb.plot_table(
                "wandb/confusion_matrix/v1",
                cm_table,
                {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"},
                {"title": "Final confusion matrix (val, pixels)"},
            ),
        }
        try:
            import matplotlib.pyplot as plt

            from src.utils.visualize import plot_confusion_matrix

            fig = plot_confusion_matrix(cm, class_names=self.class_names, normalize=True, show=False)
            log["final/confusion_matrix_normalized"] = wandb.Image(fig)
            plt.close(fig)
        except Exception as err:  # plotting is a nice-to-have
            print(f"[wandb] could not render confusion matrix figure: {err}")
        self.run.log(log)

        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            self.run.save(str(metrics_path), base_path=str(output_dir), policy="now")

        if self.wcfg.log_model:
            artifact = wandb.Artifact(
                name=f"{self.cfg.experiment_name}-unet-surrogate",
                type="model",
                metadata={"best_epoch": best.get("epoch"), **{f"final_{k}": final_val[k] for k in SCALAR_METRICS}},
            )
            for fname in ("unet_surrogate.pth", "unet_surrogate_best.pth"):
                path = output_dir / "checkpoints" / fname
                if path.exists():
                    artifact.add_file(str(path), name=fname)
            self.run.log_artifact(artifact, aliases=["latest"])

    def finish(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


def _nan_to_none(v: float) -> Optional[float]:
    return None if v != v else v


def build_tracker(
    cfg: DictConfig,
    output_dir: Path,
    class_names: Sequence[str],
    sample_images: Optional[torch.Tensor] = None,
    sample_masks: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    run_name: Optional[str] = None,
) -> LocalTracker:
    """Return the tracker selected by ``cfg.logging.backend``."""
    validate_logging_config(cfg.logging)
    if cfg.logging.backend == "wandb":
        return WandbTracker(cfg, output_dir, class_names, sample_images, sample_masks, device, run_name=run_name)
    return LocalTracker()
