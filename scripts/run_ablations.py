#!/usr/bin/env python3
"""Run the SA-LSNet ablation matrix and aggregate test metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.landslide.utils import load_config, merge_dicts, save_json  # noqa: E402

ABLATIONS = [
    {
        "name": "A0_rgb_baseline",
        "model": {"use_structural": False, "fusion": "none", "use_contrastive": False},
    },
    {
        "name": "A1_structural_concat",
        "model": {"use_structural": True, "fusion": "none", "use_contrastive": False},
    },
    {
        "name": "A2_structural_gate",
        "model": {"use_structural": True, "fusion": "gate", "use_contrastive": False},
    },
    {
        "name": "A3_structural_cross",
        "model": {"use_structural": True, "fusion": "cross", "use_contrastive": False},
    },
    {
        "name": "A4_full_sa_lsnet",
        "model": {"use_structural": True, "fusion": "gate", "use_contrastive": True},
    },
    {
        "name": "A5_contrastive_only",
        "model": {"use_structural": False, "fusion": "none", "use_contrastive": True},
    },
    {
        "name": "A6_mobilenet_light",
        "model": {
            "encoder": "mobilenet_v3_small",
            "use_structural": True,
            "fusion": "gate",
            "use_contrastive": True,
        },
    },
    {
        "name": "A7_tiny_offline",
        "model": {"encoder": "tiny", "use_structural": True, "fusion": "gate", "use_contrastive": True},
    },
]


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate(records: list[dict], out_dir: Path) -> dict:
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        grouped.setdefault(rec["run"], []).append(rec)
    keys = ["iou", "dice", "precision", "recall", "f1", "boundary_iou"]
    summary: dict[str, dict] = {}
    for name, rows in sorted(grouped.items()):
        entry: dict[str, float | int | list[float]] = {"n": len(rows)}
        for key in keys:
            vals = [float(r["metrics"][key]) for r in rows if key in r["metrics"]]
            if vals:
                entry[f"{key}_mean"] = statistics.fmean(vals)
                entry[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[name] = entry
    save_json(summary, out_dir / "summary.json")
    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        cols = ["run", "n", "f1_mean", "f1_std", "iou_mean", "dice_mean", "boundary_iou_mean"]
        f.write(",".join(cols) + "\n")
        for name, entry in sorted(summary.items()):
            f.write(
                ",".join(
                    [name]
                    + [f"{entry.get(c, ''):.6f}" if isinstance(entry.get(c), float) else str(entry.get(c, "")) for c in cols[1:]]
                )
                + "\n"
            )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("ml/configs/sa_lsnet.json"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/landslide4sense"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ablations"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="+", help="run names to include")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_config(args.config)
    base.setdefault("data", {})["root"] = str(args.data)
    selected = [a for a in ABLATIONS if not args.only or a["name"] in args.only]
    if not selected:
        raise SystemExit(f"no ablation matched {args.only}")
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for ablation in selected:
        for seed in args.seeds:
            run_name = ablation["name"]
            run_dir = args.out / run_name / f"seed{seed}"
            cfg = merge_dicts(base, {"model": ablation["model"], "seed": seed})
            cfg["experiment"] = run_name
            cfg["output"] = str(run_dir / "checkpoint")
            if args.epochs is not None:
                cfg.setdefault("train", {})["epochs"] = args.epochs
            if args.batch_size is not None:
                cfg.setdefault("data", {})["batch_size"] = args.batch_size
            if args.limit is not None:
                cfg.setdefault("data", {})["limit"] = args.limit
            if args.no_pretrained:
                cfg["model"]["pretrained"] = False
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.json"
            save_json(cfg, cfg_path)

            ckpt = run_dir / "checkpoint" / "best.pt"
            eval_path = run_dir / "eval_test.json"
            if args.skip_existing and eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                train_cmd = [
                    sys.executable,
                    "-m",
                    "ml.landslide.train",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(run_dir / "checkpoint"),
                    "--device",
                    args.device,
                ]
                run_cmd(train_cmd, args.dry_run)
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "ml.landslide.evaluate",
                    "--checkpoint",
                    str(ckpt),
                    "--data",
                    str(args.data),
                    "--split",
                    "test",
                    "--device",
                    args.device,
                    "--out",
                    str(eval_path),
                ]
                run_cmd(eval_cmd, args.dry_run)
                if args.dry_run:
                    continue
                with open(eval_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            records.append({"run": run_name, "seed": seed, "metrics": metrics})

    if args.dry_run:
        print("dry run complete")
        return
    summary = aggregate(records, args.out)
    save_json(records, args.out / "records.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
