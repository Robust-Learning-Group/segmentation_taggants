"""Datasets and DataLoaders for synthetic segmentation (wrapping ``synthdataseg``).

Two modes:

* **on-the-fly** (default): ``SyntheticSegmentationDataset`` renders every sample in ``__getitem__``.
  Rendering is CPU-heavy (~65 ms per 256x256 image), so with few CPU cores the GPU idles.
* **cached** (``cache_dir`` given): every split is rendered once (in parallel), stored as uint8 tensors in a
  ``.pt`` file and served from memory. Samples are deterministic per index (seeded), so the cached dataset is
  *identical* to the on-the-fly one -- it is purely a speed-up. Cached samples are returned as uint8
  (images ``(3, H, W)``, masks ``(H, W)``); use :func:`to_model_input` to convert batches on the device.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from synthdataseg.dataset import SyntheticSegmentationDataset
except ImportError as err:
    raise ImportError(
        "synthdataseg is required to build synthetic segmentation datasets.\n"
        "Please install the project dependencies (`uv sync`, or `pip install -r requirements.txt`)."
    ) from err

SPLIT_SEED_OFFSETS = {"train": 0, "val": 1000, "test": 2000}


def _split_seed(seed: Optional[int], split: str) -> Optional[int]:
    return None if seed is None else seed + SPLIT_SEED_OFFSETS[split]


def make_synthetic_dataset(split: str, num_samples: int, image_size: Tuple[int, int], max_n: int,
                           min_shapes: int, max_shapes: int, seed: Optional[int]) -> SyntheticSegmentationDataset:
    """On-the-fly dataset for one split (train/val/test use disjoint seeds)."""
    return SyntheticSegmentationDataset(
        num_samples=num_samples,
        image_size=tuple(image_size),
        max_n=max_n,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        seed=_split_seed(seed, split),
    )


class CachedSegmentationDataset(Dataset):
    """In-memory dataset of pre-rendered uint8 images ``(N, 3, H, W)`` and masks ``(N, H, W)``."""

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, registry: Any = None) -> None:
        if images.dtype != torch.uint8 or masks.dtype != torch.uint8:
            raise TypeError("cached images and masks must be uint8 tensors")
        if images.shape[0] != masks.shape[0]:
            raise ValueError("images and masks must have the same number of samples")
        self.images = images
        self.masks = masks
        self.registry = registry  # keeps class names / palette available like the synthetic dataset

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


def to_model_input(images: torch.Tensor, masks: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to ``device``; uint8 images become float in [0, 1], masks become int64."""
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    return images, masks.long()


# --------------------------------------------------------------------------- rendering + cache
def _render_chunk(args: Tuple[Dict[str, Any], int, int]) -> Tuple[int, np.ndarray, np.ndarray]:
    params, start, stop = args
    torch.set_num_threads(1)  # one process per core already; avoid oversubscription
    ds = make_synthetic_dataset(**params)
    n = stop - start
    h, w = params["image_size"]
    images = np.empty((n, 3, h, w), dtype=np.uint8)
    masks = np.empty((n, h, w), dtype=np.uint8)
    for k, idx in enumerate(range(start, stop)):
        img, msk = ds[idx]
        images[k] = (img * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        masks[k] = msk.numpy().astype(np.uint8)
    return start, images, masks


def render_split(params: Dict[str, Any], workers: int = 0, log_every: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render all samples of a split, in parallel over ``workers`` processes (0 = all CPU cores)."""
    n = int(params["num_samples"])
    h, w = params["image_size"]
    images = torch.empty((n, 3, h, w), dtype=torch.uint8)
    masks = torch.empty((n, h, w), dtype=torch.uint8)
    workers = workers or (os.cpu_count() or 1)
    chunk = 250
    jobs = [(params, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    done = 0
    if workers <= 1:
        results = map(_render_chunk, jobs)
    else:
        # "spawn" (not fork): forking after PyTorch has started its OpenMP threads deadlocks the children.
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(_render_chunk, jobs)
    for start, img, msk in results:
        images[start:start + len(img)] = torch.from_numpy(img)
        masks[start:start + len(msk)] = torch.from_numpy(msk)
        done += len(img)
        if done % log_every < chunk or done == n:
            print(f"  [{params['split']}] rendered {done}/{n}", flush=True)
    if workers > 1:
        pool.shutdown()
    return images, masks


def _synthdataseg_version() -> str:
    try:
        from importlib.metadata import version
        return version("synthdataseg")
    except Exception:
        return "unknown"


def cache_path(cache_dir: str | Path, params: Dict[str, Any]) -> Path:
    """Deterministic cache file name for a split's generation parameters."""
    key = dict(params, image_size=list(params["image_size"]), synthdataseg=_synthdataseg_version())
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    h, w = params["image_size"]
    name = (f"synth_{params['split']}_n{params['num_samples']}_{h}x{w}_maxn{params['max_n']}"
            f"_shapes{params['min_shapes']}-{params['max_shapes']}_seed{params['seed']}_{digest}.pt")
    return Path(cache_dir) / name


def load_or_render_split(params: Dict[str, Any], cache_dir: str | Path, workers: int = 0) -> CachedSegmentationDataset:
    """Load a split from the cache, rendering and saving it first if it does not exist yet."""
    path = cache_path(cache_dir, params)
    registry = make_synthetic_dataset(**dict(params, num_samples=1)).registry
    if path.exists():
        print(f"Loading cached {params['split']} split: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return CachedSegmentationDataset(data["images"], data["masks"], registry)

    print(f"Rendering {params['split']} split ({params['num_samples']} images) -> {path}")
    images, masks = render_split(params, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"images": images, "masks": masks, "params": params}, tmp)
    os.replace(tmp, path)  # atomic: an interrupted render never leaves a truncated cache file
    return CachedSegmentationDataset(images, masks, registry)


# --------------------------------------------------------------------------- public builders
def build_datasets(
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 8,
    train_samples: int = 8000,
    val_samples: int = 1600,
    test_samples: int = 0,
    min_shapes: int = 1,
    max_shapes: int = 4,
    max_n: Optional[int] = None,
    seed: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    generation_workers: int = 0,
) -> Dict[str, Dataset]:
    """Datasets for ``train``, ``val`` and (if ``test_samples > 0``) ``test``.

    With ``cache_dir`` the splits are rendered once and served from memory (see module docstring).
    """
    if max_n is None:
        max_n = num_classes - 1
    sizes = {"train": train_samples, "val": val_samples, "test": test_samples}
    datasets: Dict[str, Dataset] = {}
    for split, n in sizes.items():
        if n <= 0:
            continue
        params = dict(split=split, num_samples=int(n), image_size=tuple(int(v) for v in image_size),
                      max_n=int(max_n), min_shapes=int(min_shapes), max_shapes=int(max_shapes), seed=seed)
        if cache_dir:
            datasets[split] = load_or_render_split(params, cache_dir, workers=generation_workers)
        else:
            datasets[split] = make_synthetic_dataset(**params)
    return datasets


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False,
                generator: Optional[torch.Generator] = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, generator=generator)


def build_dataloaders(
    batch_size: int = 64,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 8,
    train_samples: int = 8000,
    val_samples: int = 1600,
    min_shapes: int = 1,
    max_shapes: int = 4,
    max_n: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Backward-compatible helper: on-the-fly ``(train_loader, val_loader)`` (used by the smoke-test notebook)."""
    ds = build_datasets(image_size=image_size, num_classes=num_classes, train_samples=train_samples,
                        val_samples=val_samples, min_shapes=min_shapes, max_shapes=max_shapes,
                        max_n=max_n, seed=seed)
    return (make_loader(ds["train"], batch_size, True, num_workers, pin_memory),
            make_loader(ds["val"], batch_size, False, num_workers, pin_memory))
