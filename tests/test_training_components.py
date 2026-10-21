"""Unit tests for the model, losses, augmentation, data cache and run-folder / resume logic."""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.segmentation.config import (  # noqa: E402
    ExperimentConfig, config_hash, find_resumable_run, resume_dir_from_path,
)
from src.vision.segmentation.losses import SegmentationLoss, SoftDiceLoss, random_flip_rot90  # noqa: E402
from src.vision.segmentation.models import UNetSurrogate  # noqa: E402


@pytest.mark.parametrize("depth,base,params_m", [(3, 32, 1.93), (4, 16, 1.94)])
def test_unet_depth_shapes_and_size(depth, base, params_m):
    model = UNetSurrogate(3, 6, up_channels=base, depth=depth)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 6, 64, 64)
    assert sum(p.numel() for p in model.parameters()) / 1e6 == pytest.approx(params_m, abs=0.01)


def test_unet_depth3_keeps_original_parameter_names():
    keys = set(UNetSurrogate(3, 4, 8, depth=3).state_dict())
    for name in ("down1.conv.0.weight", "down4.conv.0.weight", "up1.weight", "conv_up3.conv.0.weight", "out.weight"):
        assert name in keys
    assert not any(k.startswith("down5") for k in keys)


def test_dice_loss_perfect_and_wrong():
    target = torch.randint(0, 3, (2, 8, 8))
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    wrong = torch.nn.functional.one_hot((target + 1) % 3, 3).permute(0, 3, 1, 2).float() * 50
    dice = SoftDiceLoss(3)
    assert dice(perfect, target).item() < 1e-3
    assert dice(wrong, target).item() > 0.95
    assert SegmentationLoss(3, "ce_dice")(perfect, target).item() < 1e-3


def test_flip_rot90_moves_images_and_masks_together():
    masks = torch.randint(0, 5, (16, 12, 12))
    images = masks.unsqueeze(1).float().repeat(1, 3, 1, 1)  # image encodes the label at every pixel
    g = torch.Generator().manual_seed(0)
    aug_i, aug_m = random_flip_rot90(images, masks, generator=g)
    assert torch.equal(aug_i[:, 0].long(), aug_m)
    assert not torch.equal(aug_m, masks)  # something actually changed
    assert torch.equal(torch.bincount(aug_m.flatten(), minlength=5), torch.bincount(masks.flatten(), minlength=5))


def test_cache_roundtrip_matches_on_the_fly(tmp_path):
    from src.vision.segmentation.data import build_datasets

    kw = dict(image_size=(32, 32), num_classes=4, train_samples=3, val_samples=2, min_shapes=1, max_shapes=2, seed=7)
    fly = build_datasets(**kw)
    cached = build_datasets(**kw, cache_dir=tmp_path, generation_workers=1)
    again = build_datasets(**kw, cache_dir=tmp_path)  # loaded from disk
    for i in range(3):
        img, msk = fly["train"][i]
        for ds in (cached["train"], again["train"]):
            c_img, c_msk = ds[i]
            assert torch.allclose(c_img.float() / 255, img, atol=1 / 255 + 1e-6)
            assert torch.equal(c_msk.long(), msk)
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_config_hash_ignores_location_logging_and_resume():
    base = OmegaConf.structured(ExperimentConfig)
    other = OmegaConf.merge(base, {"results_root": "/drive", "device": "cuda", "training": {"resume": "auto"},
                                   "data": {"cache_dir": "/cache"}, "logging": {"backend": "wandb"}})
    assert config_hash(base) == config_hash(other)
    assert config_hash(base) != config_hash(OmegaConf.merge(base, {"training": {"lr": 0.002}}))


def test_find_resumable_run_and_resume_paths(tmp_path):
    exp = tmp_path / "exp"
    for name, ckpt, finished in [("20260101-000000_abc123", True, True), ("20260102-000000_abc123", True, False),
                                 ("20260103-000000_abc123", False, False), ("20260104-000000_zzz999", True, False)]:
        d = exp / name / "checkpoints"
        d.mkdir(parents=True)
        if ckpt:
            (d / "last.pth").write_bytes(b"x")
        if finished:
            (exp / name / "metrics.json").write_text("{}")
    found = find_resumable_run(tmp_path, "exp", "abc123")
    assert found is not None and found.name == "20260102-000000_abc123"  # newest unfinished with a checkpoint
    assert find_resumable_run(tmp_path, "exp", "nomatch") is None
    assert resume_dir_from_path(found / "checkpoints" / "last.pth") == found
    assert resume_dir_from_path(found) == found
    with pytest.raises(FileNotFoundError):
        resume_dir_from_path(exp / "20260103-000000_abc123")
