"""Unit tests for src.vision.segmentation.metrics against hand-computed values."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.segmentation.metrics import SegmentationMetrics  # noqa: E402


def _toy():
    # 3 classes, 2x4 image.
    target = torch.tensor([[[0, 0, 0, 0], [1, 1, 2, 2]]])
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 2, 0]]])
    return preds, target


def test_confusion_matrix_and_scalars():
    preds, target = _toy()
    m = SegmentationMetrics(num_classes=3)
    m.update(preds, target)
    r = m.compute()
    assert r["confusion_matrix"] == [[3, 1, 0], [0, 2, 0], [1, 0, 1]]
    # IoU: c0 = 3/(4+4-3)=0.6, c1 = 2/(2+3-2)=2/3, c2 = 1/(2+1-1)=0.5
    assert r["per_class_iou"] == pytest.approx([0.6, 2 / 3, 0.5])
    assert r["miou"] == pytest.approx((0.6 + 2 / 3 + 0.5) / 3)
    assert r["miou_fg"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert r["pixel_acc"] == pytest.approx(6 / 8)
    assert r["mean_acc"] == pytest.approx((3 / 4 + 1 + 1 / 2) / 3)
    # Dice: c0 = 6/8, c1 = 4/5, c2 = 2/3
    assert r["mdice"] == pytest.approx((0.75 + 0.8 + 2 / 3) / 3)
    assert r["fwiou"] == pytest.approx(0.5 * 0.6 + 0.25 * 2 / 3 + 0.25 * 0.5)


def test_logits_input_matches_labels():
    preds, target = _toy()
    logits = torch.nn.functional.one_hot(preds, 3).permute(0, 3, 1, 2).float()
    a, b = SegmentationMetrics(3), SegmentationMetrics(3)
    a.update(preds, target)
    b.update(logits, target)
    assert a.compute()["confusion_matrix"] == b.compute()["confusion_matrix"]


def test_batch_split_invariance():
    torch.manual_seed(0)
    target = torch.randint(0, 5, (6, 16, 16))
    preds = torch.randint(0, 5, (6, 16, 16))
    whole, split = SegmentationMetrics(5), SegmentationMetrics(5)
    whole.update(preds, target)
    for i in range(0, 6, 4):  # uneven batches 4 + 2
        split.update(preds[i:i + 4], target[i:i + 4])
    assert whole.compute()["miou"] == pytest.approx(split.compute()["miou"])


def test_absent_class_is_nan_and_excluded():
    target = torch.tensor([[[0, 1]]])
    preds = torch.tensor([[[0, 1]]])
    m = SegmentationMetrics(num_classes=3)
    m.update(preds, target)
    r = m.compute()
    assert math.isnan(r["per_class_iou"][2])
    assert r["miou"] == pytest.approx(1.0)


def test_ignore_index():
    target = torch.tensor([[[0, 1, 255, 255]]])
    preds = torch.tensor([[[0, 1, 0, 1]]])
    m = SegmentationMetrics(num_classes=2, ignore_index=255)
    m.update(preds, target)
    r = m.compute()
    assert r["num_pixels"] == 2
    assert r["miou"] == pytest.approx(1.0)


def test_reset():
    preds, target = _toy()
    m = SegmentationMetrics(3)
    m.update(preds, target)
    m.reset()
    assert m.compute()["num_pixels"] == 0
