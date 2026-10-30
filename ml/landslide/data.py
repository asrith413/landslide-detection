"""Dataset, deterministic grouped splits, and RGB preprocessing."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import IMAGENET_MEAN, IMAGENET_STD


def percentile_stretch(rgb: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-channel robust contrast stretch to uint8.

    This is applied identically in training and in the app, so the model sees the same
    preprocessing for Landslide4Sense patches and arbitrary UAV/phone RGB images.
    """
    rgb = rgb.astype(np.float32)
    out = np.empty_like(rgb, dtype=np.uint8)
    for c in range(rgb.shape[-1]):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, [low, high])
        if hi - lo < 1e-6:
            lo, hi = float(ch.min()), float(ch.max())
        if hi - lo < 1e-6:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any RGB image as HWC uint8 using Pillow (no OpenCV dependency)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def rgb_to_tensor(rgb_u8: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(rgb_u8)).float() / 255.0
    x = x.permute(2, 0, 1)
    m = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x - m) / s


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        image = image[::-1, :]
        mask = mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        # Photometric augmentation in uint8 space; conservative to preserve structure.
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-12, 12))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        noise = rng.normal(0, 4.0, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class Landslide4SenseDataset(Dataset):
    """RGB-only Landslide4Sense patches prepared by ``scripts/prepare_landslide4sense.py``."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        splits_file = self.root / "splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"{splits_file} not found. Run scripts/prepare_landslide4sense.py first."
            )
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split {split!r} not in {sorted(splits)}")
        self.files = [self.root / p for p in splits[split]]
        if limit:
            self.files = self.files[:limit]
        # Fixed ImageNet normalization keeps training, ONNX export, and the Rust app
        # identical, and matches the denormalization used by the structural branch.
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with np.load(self.files[index]) as data:
            image = data["image"].astype(np.uint8)
            mask = data["mask"].astype(np.uint8)
        if self.augment:
            image, mask = _augment(image, mask, self.rng)
        x = rgb_to_tensor(image, self.mean, self.std)
        y = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return {"image": x, "mask": y, "id": self.files[index].stem, "path": str(self.files[index])}


def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 16,
    augment: bool = False,
    num_workers: int = 0,
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool | None = None,
) -> DataLoader:
    ds = Landslide4SenseDataset(root, split, augment=augment, limit=limit, seed=seed)
    if shuffle is None:
        shuffle = split == "train" and augment
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Difference hash used only to group near-duplicate patches for split safety."""
    from PIL import Image

    im = Image.fromarray(image).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def grouped_split(
    items: list[dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    phash_threshold: int = 6,
) -> dict[str, list[str]]:
    """Group near-duplicate patches before splitting to reduce spatial leakage.

    ``items`` must contain ``path`` and ``hash`` (64-bit dHash) fields.  Union-find
    groups patches whose hashes are within ``phash_threshold`` bits.  Groups are then
    assigned to splits, stratified by whether they contain a landslide.
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if hamming(items[i]["hash"], items[j]["hash"]) <= phash_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    group_list = list(groups.values())
    rng = random.Random(seed)
    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
    if min(ratios.values()) < 0 or sum(ratios.values()) > 1.0 + 1e-9:
        raise ValueError("val_ratio + test_ratio must be in [0, 1]")

    def assign(stratum: list[list[int]]) -> dict[str, list[int]]:
        if not stratum:
            return {"train": [], "val": [], "test": []}
        rng.shuffle(stratum)
        total = sum(len(g) for g in stratum)
        targets = {k: total * v for k, v in ratios.items()}
        counts = {k: 0 for k in ratios}
        out = {k: [] for k in ratios}
        for g in stratum:
            name = max(ratios, key=lambda k: targets[k] - counts[k])
            out[name].extend(g)
            counts[name] += len(g)
        return out

    positive_groups = [g for g in group_list if any(items[i]["positive"] for i in g)]
    negative_groups = [g for g in group_list if not any(items[i]["positive"] for i in g)]
    split = {"train": [], "val": [], "test": []}
    for stratum in (positive_groups, negative_groups):
        assigned = assign(stratum)
        for name in split:
            split[name].extend(assigned[name])

    for name in ("val", "test"):
        if not split[name]:
            if not split["train"]:
                raise ValueError("not enough groups to create non-empty splits")
            split[name].append(split["train"].pop())
    out = {k: sorted(items[i]["path"] for i in v) for k, v in split.items()}
    return out


def stats_from_files(paths: Iterable[Path], max_images: int = 500) -> dict[str, list[float]]:
    paths = list(paths)[:max_images]
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for p in paths:
        with np.load(p) as d:
            x = d["image"].astype(np.float32) / 255.0
        total += x.reshape(-1, 3).sum(0)
        total_sq += (x.reshape(-1, 3) ** 2).sum(0)
        count += x.shape[0] * x.shape[1]
    mean = total / max(count, 1)
    var = np.maximum(total_sq / max(count, 1) - mean**2, 1e-8)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}
