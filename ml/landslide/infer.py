#!/usr/bin/env python3
"""Python reference inference with tiled overlap blending (parity with the app)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import load_rgb, percentile_stretch, rgb_to_tensor
from .model import build_model
from .tiling import extract_replicated, stitch_probs, tile_positions


@torch.no_grad()
def predict_array(
    model: torch.nn.Module,
    rgb: np.ndarray,
    tile: int = 128,
    overlap: int = 32,
    batch_size: int = 1,
    device: str = "cpu",
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability map, binary mask) for an HWC uint8 RGB image.

    The default batch size is 1 because PyTorch CPU kernels can produce slightly
    different numerics for different batch sizes; batch 1 matches the Rust app.
    """
    dev = torch.device(device)
    rgb = percentile_stretch(rgb)
    height, width = rgb.shape[:2]
    positions = tile_positions(height, width, tile, overlap)
    tiles = [
        rgb_to_tensor(extract_replicated(rgb, t.y, t.x, height, width, tile))
        for t in positions
    ]
    probs: list[tuple] = []
    model.eval()
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack(tiles[i : i + batch_size]).to(dev)
        p = torch.sigmoid(model(batch)["logits"]).cpu().numpy()[:, 0]
        for j, t in enumerate(positions[i : i + batch_size]):
            probs.append((t, p[j][: t.h, : t.w]))
    full = stitch_probs(probs, height, width, overlap)
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
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    rgb = load_rgb(args.image)
    prob, mask = predict_array(
        model,
        rgb,
        args.tile,
        args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    stem = args.image.stem
    Image.fromarray(mask * 255).save(args.out / f"{stem}_mask.png")
    Image.fromarray((prob * 255).astype(np.uint8)).save(args.out / f"{stem}_prob.png")
    Image.fromarray(overlay(percentile_stretch(rgb), prob)).save(args.out / f"{stem}_overlay.png")
    print(f"wrote {args.out}/{stem}_{{mask,prob,overlay}}.png")


if __name__ == "__main__":
    main()
