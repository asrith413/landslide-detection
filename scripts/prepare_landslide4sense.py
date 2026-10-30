#!/usr/bin/env python3
"""Convert the official 14-band Landslide4Sense HDF5 release to RGB-only NPZ patches.

The official test masks are withheld by the organisers.  We combine the labelled
training and validation releases (4,044 patches) and create a deterministic,
near-duplicate-grouped 70/15/15 split.  Never claim official test-set numbers.

Example
-------
python scripts/prepare_landslide4sense.py \
    --raw data/raw --out data/processed/landslide4sense
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from landslide.data import dhash, grouped_split, percentile_stretch, stats_from_files  # noqa: E402

RGB_BANDS = (3, 2, 1)  # Sentinel-2 B4 (red), B3 (green), B2 (blue), 1-based -> 0-based


def find_h5(root: Path, kind: str) -> list[Path]:
    for pattern in (f"**/{kind}/*.h5", f"**/{kind}/*.hdf5"):
        files = sorted(root.glob(pattern))
        if files:
            return files
    return []


def maybe_extract(raw: Path, out: Path) -> None:
    """Extract TrainData.zip / ValidData.zip if needed."""
    for name in ("TrainData.zip", "ValidData.zip"):
        zpath = raw / name
        if not zpath.exists():
            continue
        target = raw / "extracted" / name.replace(".zip", "")
        marker = target / ".extracted"
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"extracting {zpath} -> {target}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
        marker.write_text("ok\n")


def read_pair(img_path: Path, mask_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(img_path, "r") as h:
        key = "img" if "img" in h else list(h.keys())[0]
        arr = np.asarray(h[key][:], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(f"unexpected image shape {arr.shape} in {img_path}")
    rgb = arr[..., list(RGB_BANDS)]
    rgb_u8 = percentile_stretch(rgb)
    if mask_path is None:
        mask = np.zeros(rgb_u8.shape[:2], dtype=np.uint8)
    else:
        with h5py.File(mask_path, "r") as h:
            key = "mask" if "mask" in h else list(h.keys())[0]
            mask = np.asarray(h[key][:])
        mask = (mask > 0.5).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != rgb_u8.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} != image {rgb_u8.shape} in {mask_path}")
    return rgb_u8, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/landslide4sense"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--no-phash", action="store_true", help="disable near-duplicate grouping")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    maybe_extract(args.raw, args.out)

    img_files = find_h5(args.raw, "img")
    mask_files = find_h5(args.raw, "mask")
    if not img_files:
        raise SystemExit(
            f"No image HDF5 files found under {args.raw}. Download TrainData.zip from "
            "https://doi.org/10.5281/zenodo.10463239 and re-run."
        )
    mask_by_stem = {p.stem.replace("mask_", ""): p for p in mask_files}
    pairs: list[tuple[Path, Path | None]] = []
    for p in img_files:
        stem = p.stem.replace("image_", "")
        pairs.append((p, mask_by_stem.get(stem)))
    labelled = [(p, m) for p, m in pairs if m is not None]
    if not labelled:
        raise SystemExit("No image/mask pairs found; cannot train a supervised model.")
    if args.limit:
        labelled = labelled[: args.limit]
    print(f"found {len(pairs)} images, {len(labelled)} labelled pairs")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    tmp_dir = args.out / "_tmp"
    tmp_dir.mkdir()
    for i, (img_path, mask_path) in enumerate(labelled):
        rgb, mask = read_pair(img_path, mask_path)
        out_path = tmp_dir / f"patch_{i:05d}.npz"
        np.savez(out_path, image=rgb, mask=mask)
        items.append(
            {
                "path": f"patches/patch_{i:05d}.npz",
                "hash": dhash(rgb),
                "positive": bool(mask.any()),
                "source": img_path.name,
            }
        )
        if (i + 1) % 250 == 0:
            print(f"  processed {i + 1}/{len(labelled)}")

    if args.no_phash:
        # Hash every item to a unique value so no grouping happens.
        for i, it in enumerate(items):
            it["hash"] = i
        threshold = -1
    else:
        threshold = args.phash_threshold
    splits = grouped_split(
        items,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        phash_threshold=max(threshold, 0),
    )

    patches = args.out / "patches"
    tmp_dir.rename(patches)
    for name in ("train", "val", "test"):
        n_pos = sum(
            1
            for p in splits[name]
            if next(it for it in items if it["path"] == p)["positive"]
        )
        print(f"{name}: {len(splits[name])} patches, {n_pos} with landslide")

    (args.out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    stats = stats_from_files([args.out / p for p in splits["train"]])
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    info = {
        "dataset": "Landslide4Sense",
        "source": "https://doi.org/10.5281/zenodo.10463239",
        "license": "CC BY 4.0",
        "rgb_bands": ["B4 (red)", "B3 (green)", "B2 (blue)"],
        "preprocessing": "per-channel 2nd-98th percentile stretch to uint8",
        "split": {
            "method": "near-duplicate-grouped dHash split" if not args.no_phash else "random",
            "phash_threshold": None if args.no_phash else args.phash_threshold,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "note": "official test labels are withheld; these splits come from the labelled train release",
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "rgb_stats": stats,
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
