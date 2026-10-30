#!/usr/bin/env python3
"""Export a deployment ONNX model returning a single sigmoid probability map."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import DeploymentModel
from .utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/sa_lsnet.onnx"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--dynamo",
        action="store_true",
        help="use the torch.export-based exporter (requires onnxscript)",
    )
    args = ap.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ONNX export requires the optional 'onnx' package: pip install onnx"
        ) from exc

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from .model import build_model

    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    deploy = DeploymentModel(model).eval()
    dummy = torch.randn(1, 3, args.size, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            deploy,
            dummy,
            str(args.out),
            input_names=["image"],
            output_names=["probability"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=args.dynamo,
        )
    meta = {
        "checkpoint": str(args.checkpoint),
        "input": {"name": "image", "shape": [1, 3, args.size, args.size], "layout": "NCHW"},
        "output": {"name": "probability", "shape": [1, 1, args.size, args.size]},
        "normalization": {
            "rgb": "per-channel 2nd-98th percentile stretch to uint8, then ImageNet mean/std",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "threshold": 0.5,
        "tile_size": args.size,
        "recommended_overlap": 32,
        "config": ckpt["config"],
    }
    save_json(meta, args.out.with_suffix(".json"))
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
