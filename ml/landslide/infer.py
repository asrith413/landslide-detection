#!/usr/bin/env python3
"""Python reference inference with tiled overlap blending (parity with the app)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import load_rgb, percentile_stretch, rgb_to_tensor
from .model import build_model
from .tiling import pad_to_tile, stitch_probs, tile_positions, unpad


@torch.no_grad()
def predict_array(
    model: torch.nn.Module,
    rgb: np.ndarray,
    tile: int = 128,
    overlap: int = 32,
    batch_size: int = 8,
    device: str = "cpu",
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability map, binary mask) for an HWC uint8 RGB image."""
    dev = torch.device(device)
    rgb = percentile_stretch(rgb)
    padded, orig = pad_to_tile(rgb, tile)
    positions = tile_positions(padded.shape[0], padded.shape[1], tile, overlap)
    tiles = []
    for t in positions:
        patch = padded[t.y : t.y + t.h, t.x : t.x + t.w]
        if patch.shape[0] != tile or patch.shape[1] != tile:
            canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
            canvas[: patch.shape[0], : patch.shape[1]] = patch
            patch = canvas
        tiles.append(rgb_to_tensor(patch))
    probs: list[tuple] = []
    model.eval()
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack(tiles[i : i + batch_size]).to(dev)
        p = torch.sigmoid(model(batch)["logits"]).cpu().numpy()[:, 0]
        for j, t in enumerate(positions[i : i + batch_size]):
            probs.append((t, p[j][: t.h, : t.w]))
    full = stitch_probs(probs, padded.shape[0], padded.shape[1], overlap)
    full = unpad(full, orig)
    return full, (full >= threshold).astype(np.uint8)


def overlay(rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    color[..., 1] = 40
    mask = (prob >= 0.5).astype(np.float32)[..., None]
    return (rgb.astype(np.float32) * (1 - alpha * mask) + color * alpha * mask).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/infer"))
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    rgb = load_rgb(args.image)
    prob, mask = predict_array(model, rgb, args.tile, args.overlap, device=args.device, threshold=args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    stem = args.image.stem
    Image.fromarray(mask * 255).save(args.out / f"{stem}_mask.png")
    Image.fromarray((prob * 255).astype(np.uint8)).save(args.out / f"{stem}_prob.png")
    Image.fromarray(overlay(percentile_stretch(rgb), prob)).save(args.out / f"{stem}_overlay.png")
    print(f"wrote {args.out}/{stem}_{{mask,prob,overlay}}.png")


if __name__ == "__main__":
    main()
