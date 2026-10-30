#!/usr/bin/env python3
"""Evaluate a checkpoint with segmentation, boundary, and efficiency metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import build_dataloader
from .losses import BCEDiceLoss, MaskedObjectContrastiveLoss, TotalLoss
from .metrics import SegmentationMetrics
from .model import build_model
from .utils import count_parameters, load_json, model_size_mb, peak_memory_mb, save_json


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def run(
    checkpoint: Path,
    data_root: Path,
    split: str = "test",
    device: str = "auto",
    tta: bool = False,
    limit: int | None = None,
    out: Path | None = None,
) -> dict:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model, cfg, ckpt = load_checkpoint(checkpoint, dev)
    loader = build_dataloader(
        data_root,
        split,
        batch_size=1,
        augment=False,
        num_workers=0,
        limit=limit,
    )
    metrics = SegmentationMetrics()
    latencies: list[float] = []
    for batch in loader:
        x = batch["image"].to(dev)
        y = batch["mask"].to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)["logits"]
        if tta:
            for flip in (2, 3):
                logits = logits + torch.flip(model(torch.flip(x, dims=[flip]))["logits"], dims=[flip])
            logits = logits / 3.0
        if dev.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        metrics.update(logits, y, compute_boundary=True)
    result = metrics.compute()
    # Warm-up + repeated single-image latency.
    sample = next(iter(loader))["image"].to(dev)
    for _ in range(3):
        model(sample)
    reps = 20
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(sample)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    result.update(
        {
            "split": split,
            "checkpoint": str(checkpoint),
            "best_epoch": ckpt.get("epoch"),
            "parameters": count_parameters(model),
            "model_size_mb": model_size_mb(model),
            "latency_ms_per_image": float(np.mean(latencies)),
            "latency_ms_mean_20": (time.perf_counter() - t0) * 1000.0 / reps,
            "peak_rss_mb": peak_memory_mb(),
            "device": str(dev),
            "input_size": list(sample.shape),
            "tta": tta,
            "config": cfg,
        }
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
    if out:
        save_json(result, out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/landslide4sense"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run(args.checkpoint, args.data, args.split, args.device, args.tta, args.limit, args.out)


if __name__ == "__main__":
    main()
