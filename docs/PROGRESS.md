# Progress

Last updated: 2026-10-30

This file separates what is implemented and verified from what is planned. Do not
promote an item to "done" without a command or artifact that proves it.

## Status summary

| Area | Status |
|---|---|
| Python package (model, losses, data, metrics, tiling) | implemented |
| Unit and smoke tests | 13 passing |
| Dataset download | in progress (Zenodo, parallel segmented) |
| Dataset preprocessing | implemented, not yet run on full data |
| Training | implemented, smoke run pending data |
| Ablations | runner implemented, dry-run verified |
| ONNX export | implemented, not yet tested |
| Rust inference core | implemented, 11 tests passing |
| Tauri app | planned |
| Docs | architecture, literature review, experimental plan, progress |
| Git history / push | 2 commits pushed to private `main` |

## Verified

Environment (Termux, aarch64):

- Python 3.14.8, PyTorch 2.14.1, torchvision 0.29.2, NumPy 2.5.4, Pillow 13.0.0,
  SciPy, h5py 3.16.0, Rust 1.99.1, Node 26.12.1, git 2.56.0, gh 2.103.3.
- No GPU, no Android SDK/NDK, no webkit2gtk.

Model checks:

- `regnet_y_400mf` encoder channels `(48, 104, 208, 440)` at strides 4/8/16/32.
- `mobilenet_v3_small` encoder channels `(16, 24, 40, 96)`.
- Full regnet model: 12,491,741 parameters; mobilenet light: 3,578,434.
- Forward outputs: logits `(B,1,128,128)`, embedding `(B,64,32,32)`.
- Backward pass produces finite gradients.

Tests:

```bash
python3 -m unittest discover -s tests -v
# Ran 13 tests ... OK
```

Data source checks:

- HF mirror `ibm-nasa-geospatial/Landslide4sense`: annotation HDF5 files are empty
  (about 800 bytes, no datasets); parquet contains image bytes plus a split label, not
  masks; `val_0.parquet` is corrupt. Not usable for supervised training.
- Official Zenodo release is required. TrainData.zip is 2,484,071,934 bytes.

Rust core (`crates/landslide-core`):

- `tract-onnx` 0.23.9, no ONNX Runtime.
- Tiling, overlap blending, percentile stretch, ImageNet normalization, tiled
  inference, RGBA overlay, and JSON history.
- `cargo test --workspace`: 11 passed.

Ablation runner:

- `scripts/run_ablations.py` covers A0-A7 from `docs/EXPERIMENTAL_PLAN.md`.
- Dry-run verified for A0.
- Writes per-run configs, checkpoints, test metrics, `summary.json`, `summary.csv`.

Fixes completed:

- Entropy one-hot channel squeeze.
- Contrastive mask channel squeeze.
- Replicate padding in structural features.
- MobileNet duplicate-scale collection.
- Tiling ramp clamp for tiles smaller than `2*overlap`; regression test added.
- README config extension corrected from `.yaml` to `.json`.

## In progress

Zenodo download (detached, 12 parallel range requests):

- `data/raw/TrainData.zip.parts/` was about 1.87 GB of 2.48 GB at 2026-10-30 16:28
  IST after resuming from partial segments.
- `ValidData.zip` (143,423,314 bytes) is queued after TrainData.
- Completion marker: `data/raw/.download_done`.
- Log: `data/raw/par_dl.log`.
- Helper scripts: `/data/data/com.termux/files/usr/tmp/segdl.py`,
  `segdl_worker.py`, `start_dl.sh`.

Monitor:

```bash
du -sb data/raw/TrainData.zip.parts
ls data/raw/.download_done
```

## Next steps

1. Wait for download completion, verify the zip integrity, extract.
2. Run `scripts/prepare_landslide4sense.py` to build RGB NPZ files, grouped splits,
   and statistics.
3. Run a 64-image CPU smoke training to confirm the full loop.
4. Export ONNX and test `onnx` availability; add a parity check.
5. Implement `scripts/run_ablations.py` to execute the matrix in
   `docs/EXPERIMENTAL_PLAN.md` and write `outputs/`.
6. Implement `crates/landslide-core` with `tract-onnx` (tiling, blending, overlay,
   history).
7. Scaffold the Tauri 2 app (upload, preview, mask, overlay, timing, export, history).
8. Initialize git, commit in reviewable chunks, push to the private GitHub repo.

## Planned, not started

- `crates/landslide-core` and `app/`.
- `scripts/run_ablations.py`.
- Cross-region evaluation on Bijie/UAV data.
- Android build instructions and CI (cannot build APK in this environment).
- Licence file (needs a decision).
- Foundation-model encoder experiment (future work).

## Known issues and risks

- Full training on this CPU is not practical; use the Windows CUDA laptop.
- ONNX export is untested; `dynamo=False` may require the `onnx` package.
- The official Landslide4Sense test masks are withheld; all internal test numbers come
  from the grouped hold-out and must be labelled as such.
- Cross-region generalization is unproven until a labelled target dataset is used.
- Contrastive loss is skipped on batches without both classes; monitor its effective
  batch coverage.
- No licence file yet; do not publish results or weights until this is decided.

## Commands

```bash
# python tests
python3 -m unittest discover -s tests -v

# rust tests
cargo test --workspace

# ablation dry run
python3 scripts/run_ablations.py --seeds 42 --dry-run

# prepare data (after extraction)
python3 scripts/prepare_landslide4sense.py \
  --raw data/raw --out data/processed/landslide4sense

# smoke train (CPU, no pretrained weights)
python3 -m ml.landslide.train \
  --config ml/configs/rgb_baseline.json \
  --epochs 1 --limit 64 --batch-size 4 --no-pretrained

# evaluate
python3 -m ml.landslide.evaluate \
  --checkpoint checkpoints/sa_lsnet/best.pt \
  --data data/processed/landslide4sense --split test

# export
python3 -m ml.landslide.export_onnx \
  --checkpoint checkpoints/sa_lsnet/best.pt --out models/sa_lsnet.onnx
```
