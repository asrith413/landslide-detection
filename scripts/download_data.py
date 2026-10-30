#!/usr/bin/env python3
"""Download public landslide datasets with explicit provenance.

Only the official Landslide4Sense Zenodo release is used for the primary experiments.
The Hugging Face mirror is intentionally *not* used for masks: its annotation HDF5
files were verified to be empty (all files share one 800-byte object with no datasets).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ZENODO = {
    "TrainData.zip": "https://zenodo.org/records/10463239/files/TrainData.zip?download=1",
    "ValidData.zip": "https://zenodo.org/records/10463239/files/ValidData.zip?download=1",
}


def curl(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"exists: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {url}\n -> {out}")
    subprocess.run(["curl", "-L", "--retry", "5", "--retry-delay", "5", "-o", str(out), url], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="landslide4sense", choices=["landslide4sense"])
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--valid-only", action="store_true")
    args = ap.parse_args()
    if args.dataset == "landslide4sense":
        names = ["ValidData.zip"] if args.valid_only else list(ZENODO)
        for name in names:
            curl(ZENODO[name], args.out / name)
        print("Next: python scripts/prepare_landslide4sense.py --raw data/raw")
    else:
        sys.exit(f"unsupported dataset {args.dataset}")


if __name__ == "__main__":
    main()
