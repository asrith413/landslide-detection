"""Overlap-blended tiled inference for large RGB images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    y: int
    x: int
    h: int
    w: int


def tile_positions(height: int, width: int, tile: int = 128, overlap: int = 32) -> list[Tile]:
    if tile <= 0:
        raise ValueError("tile must be positive")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be in [0, tile)")
    stride = tile - overlap

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        vals = list(range(0, total - tile + 1, stride))
        if vals[-1] != total - tile:
            vals.append(total - tile)
        return vals

    return [
        Tile(y, x, min(tile, height - y), min(tile, width - x))
        for y in starts(height)
        for x in starts(width)
    ]


def _blend_window(h: int, w: int, overlap: int) -> np.ndarray:
    """Separable linear ramp; overlapping ramps sum to one away from borders."""
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    ov_y = min(overlap, max((h - 1) // 2, 0))
    ov_x = min(overlap, max((w - 1) // 2, 0))
    if ov_y > 0:
        ramp = np.linspace(1.0 / (ov_y + 1), 1.0, ov_y, dtype=np.float32)
        wy[:ov_y] = ramp
        wy[-ov_y:] = ramp[::-1]
    if ov_x > 0:
        ramp = np.linspace(1.0 / (ov_x + 1), 1.0, ov_x, dtype=np.float32)
        wx[:ov_x] = ramp
        wx[-ov_x:] = ramp[::-1]
    return np.outer(wy, wx)


def stitch_probs(
    tiles: list[tuple[Tile, np.ndarray]],
    height: int,
    width: int,
    overlap: int = 32,
) -> np.ndarray:
    """Blend tile probability maps into a full-size map in [0, 1]."""
    acc = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for t, prob in tiles:
        p = np.asarray(prob, dtype=np.float32)
        if p.ndim == 3:
            p = p[0]
        p = p[: t.h, : t.w]
        win = _blend_window(p.shape[0], p.shape[1], overlap)
        acc[t.y : t.y + p.shape[0], t.x : t.x + p.shape[1]] += p * win
        weight[t.y : t.y + p.shape[0], t.x : t.x + p.shape[1]] += win
    return np.clip(acc / np.maximum(weight, 1e-6), 0.0, 1.0)


def pad_to_tile(image: np.ndarray, tile: int = 128) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    ph = max(tile, int(np.ceil(h / tile) * tile))
    pw = max(tile, int(np.ceil(w / tile) * tile))
    if (ph, pw) == (h, w):
        return image, (h, w)
    if image.ndim == 2:
        out = np.zeros((ph, pw), dtype=image.dtype)
        out[:h, :w] = image
    else:
        out = np.zeros((ph, pw, image.shape[2]), dtype=image.dtype)
        out[:h, :w] = image
    return out, (h, w)


def unpad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return image[:h, :w]
