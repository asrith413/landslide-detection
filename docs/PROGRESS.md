# Progress

Last updated: 2026-10-30

This file separates what is implemented and verified from what is planned. Do not
promote an item to "done" without a command or artifact that proves it.

## Status summary

| Area | Status |
|---|---|
| Python package (model, losses, data, metrics, tiling) | implemented |
| Unit and smoke tests | 16 passing |
| Dataset download | complete (Zenodo, verified extraction) |
| Dataset preprocessing | complete: 4044 labelled RGB patches |
| Training | implemented; 1-epoch CPU smoke passed |
| Ablations | runner implemented, dry-run verified |
| ONNX export | implemented, not yet tested |
| Rust inference core | implemented, 11 tests passing |
| Tauri app | scaffolded; frontend builds; native build blocked by SDK/webkit |
| Docs | architecture, literature review, experimental plan, progress |
| Git history / push | pushed to private `main` |

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
# Ran 16 tests ... OK
```

Data source checks:

- HF mirror `ibm-nasa-geospatial/Landslide4sense`: annotation HDF5 files are empty
  (about 800 bytes, no datasets); parquet contains image bytes plus a split label, not
  masks; `val_0.parquet` is corrupt. Not usable for supervised training.
- Official Zenodo release downloaded: TrainData.zip 2,484,071,934 bytes,
  ValidData.zip 143,423,314 bytes.
- Extracted 3,799 train + 245 validation patches = 4,044 labelled pairs.
- Prepared 4,044 RGB NPZ patches (269 MB) under `data/processed/landslide4sense`.
- Grouped splits: train 2,831 (1,639 positive), val 607 (351 positive),
  test 606 (351 positive).
- Dataset RGB stats recorded in `stats.json` (mean `[0.2794, 0.3535, 0.2887]`,
  std `[0.2439, 0.2478, 0.2425]`); training, export, and Rust use fixed ImageNet
  mean/std so the structural branch and deployment preprocessing stay consistent.

Smoke training and evaluation:

- 1 epoch, 64 train patches, full 607-patch validation, CPU.
- Completed in 133.7 s; val Dice 0.0010. This only proves the training loop runs;
  it is not a meaningful model-quality result.
- Evaluation CLI ran on 32 test patches with boundary metrics enabled: IoU 0.0,
  Dice 0.0, Boundary IoU 0.2236, ~286 ms/image, 431 MB peak RSS on CPU.
  These are pipeline checks, not reportable model results.

Rust core (`crates/landslide-core`):

- `tract-onnx` 0.23.9, no ONNX Runtime.
- Tiling, overlap blending, percentile stretch, ImageNet normalization, tiled
  inference, RGBA overlay, and JSON history.
- `cargo test --workspace`: 11 passed.

Ablation runner:

- `scripts/run_ablations.py` covers A0-A7 from `docs/EXPERIMENTAL_PLAN.md`.
- Dry-run verified for A0.
- Writes per-run configs, checkpoints, test metrics, `summary.json`, `summary.csv`.

Tauri app scaffold:

- `app/` with Vite + TypeScript UI, model/image pickers, threshold control, three
  views, metrics, and local history.
- `app/src-tauri/` exposes a `segment_image` command backed by `landslide-core`.
- Frontend verified with `tsc --noEmit` and `vite build`.
- Native shell not built here: no Android SDK/NDK and no desktop webkit libraries.
- `landslide-cli` binary added for batch inference and deployment testing.

Fixes completed:

- Entropy one-hot channel squeeze.
- Contrastive mask channel squeeze.
- Replicate padding in structural features.
- MobileNet duplicate-scale collection.
- Tiling ramp clamp for tiles smaller than `2*overlap`; regression test added.
- README config extension corrected from `.yaml` to `.json`.
- Grouped split made stratified: previous ordering put only positives in val/test;
  now each split contains both classes. Regression test added.
- Boundary metrics now squeeze the singleton channel before pooling; regression test
  added for `(B,1,H,W)` accumulator input.
- Boundary IoU returns 1.0 for identical masks with empty boundary bands.
- Peak RSS unit corrected for Linux/Termux (KiB, not bytes).
- Dataset normalization fixed to ImageNet mean/std to match `denormalize_rgb`, ONNX
  metadata, and `landslide-core`; dataset-specific stats were previously applied while
  the structural branch denormalized with ImageNet values.

## In progress

- ONNX export and Rust/PyTorch parity are the next unverified links.
- `python3 -m pip install onnx` did not complete in this Termux environment; export
  may need to run on the Windows machine or use a prebuilt aarch64 wheel.

## Next steps

1. Export a checkpoint to ONNX (opset 17, fixed 128 input) and verify with `onnx`.
2. Run `landslide-cli` on a prepared patch and compare mask/IoU against PyTorch.
3. Build the Tauri desktop app on Windows; keep Android build for later.
4. Train A0/A1 baselines and SA-LSNet on the CUDA laptop with the documented seeds.
5. Run the A0-A7 ablation matrix and write measured results into the docs.
6. Cross-region evaluation on Bijie/UAV data if labelled target data is available.

## Planned, not started

- Cross-region evaluation on Bijie/UAV data.
- Android build instructions and CI (cannot build APK in this environment).
- Licence file (needs a decision).
- Foundation-model encoder experiment (future work).

## Known issues and risks

- Full training on this CPU is not practical; use the Windows CUDA laptop.
- ONNX export is untested; `dynamo=False` requires the `onnx` package, which is not installed here.
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
  --epochs 1 --limit 64 --batch-size 4 --no-pretrained \
  --no-boundary --output outputs/smoke

# evaluate
python3 -m ml.landslide.evaluate \
  --checkpoint checkpoints/sa_lsnet/best.pt \
  --data data/processed/landslide4sense --split test

# quick evaluation smoke
python3 -m ml.landslide.evaluate \
  --checkpoint outputs/smoke/best.pt \
  --data data/processed/landslide4sense --split test --limit 32

# export
python3 -m ml.landslide.export_onnx \
  --checkpoint checkpoints/sa_lsnet/best.pt --out models/sa_lsnet.onnx
```
