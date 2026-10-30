#!/usr/bin/env python3
"""Compare Python PyTorch inference with the Rust tract-onnx CLI.

Exact parity requires single-threaded, batch-1 PyTorch inference because CPU
convolution kernels can change numerics with batch size and thread count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from landslide.data import load_rgb  # noqa: E402
from landslide.infer import predict_array  # noqa: E402
from landslide.model import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "parity")
    ap.add_argument("--cli", type=Path, help="path to a built landslide-cli")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-diff", type=int, default=1, help="max 8-bit probability diff")
    ap.add_argument("--min-agreement", type=float, default=0.999)
    args = ap.parse_args()

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rgb = load_rgb(args.image)
    prob, mask = predict_array(
        model,
        rgb,
        tile=args.tile,
        overlap=args.overlap,
        batch_size=1,
        device="cpu",
        threshold=args.threshold,
    )
    py_prob = np.clip((prob * 255.0).astype(np.uint8), 0, 255)
    Image.fromarray(py_prob).save(args.out / "pytorch_prob.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(args.out / "pytorch_mask.png")

    rust_prob_path = args.out / "rust_prob.png"
    rust_mask_path = args.out / "rust_mask.png"
    if args.cli:
        command = [str(args.cli)]
    else:
        command = ["cargo", "run", "--quiet", "--bin", "landslide-cli", "--"]
    command += [
        "--model",
        str(args.onnx),
        "--input",
        str(args.image),
        "--mask",
        str(rust_mask_path),
        "--prob",
        str(rust_prob_path),
        "--threshold",
        str(args.threshold),
        "--tile",
        str(args.tile),
        "--overlap",
        str(args.overlap),
    ]
    run = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    rust_summary = json.loads(run.stdout.strip().splitlines()[-1])

    rust_prob = np.asarray(Image.open(rust_prob_path), dtype=np.int16)
    rust_mask = np.asarray(Image.open(rust_mask_path)) > 0
    diff = np.abs(rust_prob - py_prob.astype(np.int16))
    inter = int((mask & rust_mask).sum())
    union = int((mask | rust_mask).sum())
    py_sum = int(mask.sum())
    rust_sum = int(rust_mask.sum())
    summary = {
        "image": str(args.image),
        "tiles": rust_summary["tiles"],
        "probability_max_abs_diff_8bit": int(diff.max()),
        "probability_mean_abs_diff_8bit": float(diff.mean()),
        "probability_fraction_within_1": float((diff <= 1).mean()),
        "mask_agreement": float((mask == rust_mask).mean()),
        "mask_iou": float(inter / union) if union else 1.0,
        "mask_dice": float(2 * inter / (py_sum + rust_sum)) if (py_sum + rust_sum) else 1.0,
        "python_coverage": float(mask.mean()),
        "rust_coverage": float(rust_mask.mean()),
        "rust_latency_ms": rust_summary["latency_ms"],
    }
    (args.out / "parity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if diff.max() > args.max_diff:
        raise SystemExit(f"probability mismatch: max 8-bit diff {int(diff.max())} > {args.max_diff}")
    if summary["mask_agreement"] < args.min_agreement:
        raise SystemExit(
            f"mask mismatch: agreement {summary['mask_agreement']:.6f} < {args.min_agreement}"
        )


if __name__ == "__main__":
    main()
