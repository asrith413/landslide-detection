# Progress

Last updated: 2026-10-30

This file separates what is implemented and verified from what is planned. Do not
promote an item to "done" without a command or artifact that proves it.

## Status summary

| Area | Status |
|---|---|
| Python package (model, losses, data, metrics, tiling) | implemented |
| Unit and smoke tests | 18 passing |
| Dataset download | complete (Zenodo, verified extraction) |
| Dataset preprocessing | complete: 4044 labelled RGB patches |
| Training | implemented; RGB and SA-LSNet CPU smokes passed |
| Ablations | runner implemented, dry-run verified |
| ONNX export | verified: opset 18 dynamo export works |
| Rust inference core | implemented, 12 tests passing |
| Deployment parity | verified: PyTorch vs Rust exact on 1-tile and 6-tile images |
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
- Full regnet model: 12,491,741 parameters; mobilenet light: 3,321,450.
- Forward outputs: logits `(B,1,128,128)`, embedding `(B,64,32,32)`.
- Backward pass produces finite gradients.
- `DeploymentModel` returns a `(1,1,128,128)` probability map in `[0,1]`.

Tests:

```bash
python3 -m unittest discover -s tests -v
# Ran 18 tests ... OK
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
- Completed in 114.3 s; val Dice 0.0396. This only proves the training loop runs;
  it is not a meaningful model-quality result.
- Evaluation CLI ran on 32 test patches with boundary metrics enabled: IoU 0.0390,
  Dice 0.0751, Boundary IoU 0.1308, ~1.13 s/image, 378 MB peak RSS on CPU.
  These are pipeline checks, not reportable model results.
- Full SA-LSNet path also smoke-tested: 1 epoch, 16 train / 32 val patches, batch 2,
  no pretrained weights, CPU. Completed in 88.9 s; val Dice 0.0415. This exercises
  structural features, gated fusion, and contrastive loss, not just the RGB baseline.

Rust core (`crates/landslide-core`):

- `tract-onnx` 0.23.9, no ONNX Runtime.
- Tiling, overlap blending, percentile stretch, ImageNet normalization, tiled
  inference, RGBA overlay, and JSON history.
- `cargo test --workspace`: 12 passed.

Deployment parity:

- `onnx 1.23.2` built from source in Termux and relinked against `libpython3.14.so`.
- `python3 -m ml.landslide.export_onnx ... --dynamo` writes a fixed 128x128 opset-18
  model with external data; `tract-onnx` 0.23.9 loads it.
- `scripts/check_parity.py` runs single-threaded, batch-1 PyTorch inference and the Rust
  CLI on the same image. On the smoke checkpoint: 1-tile 128x128 image and 300x200
  6-tile image both give max 8-bit probability difference 0 and mask IoU/Dice 1.0.
- The same export/parity path was verified with random-weight full SA-LSNet and
  MobileNet-light checkpoints, so the structural features, gated fusion, and MobileNet
  ops are supported by `tract-onnx` 0.23.9.
- The 1-epoch SA-LSNet smoke checkpoint exports and runs through `landslide-cli` with
  max 8-bit probability difference 1, mean 0.0000167, and mask IoU/Dice 1.0 on a
  300x200 6-tile image.
- PyTorch CPU batch size and thread count change numerics slightly, so parity checks
  force `torch.set_num_threads(1)` and `batch_size=1`; the Rust app processes one tile
  at a time.

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
- Orientation features now use algebraic `cos(2θ)`/`sin(2θ)` instead of
  `atan2`+`cos`/`sin`, improving ONNX/tract export compatibility.
- Percentile stretch is applied once to the full image before tiling in both Python and
  Rust; edge tiles use replicate padding. The previous Python path padded to a tile
  multiple and used zero padding, which did not match the app on non-tile-size images.
- Reference inference defaults to batch size 1 so it matches the Rust single-tile path.

## In progress

- ONNX export is working in Termux after building `onnx 1.23.2` from source.
- The PyTorch 2.14 legacy exporter still fails with
  `required keyword attribute 'value' has the wrong type`; `--dynamo` with `onnxscript`
  succeeds and writes opset 18.
- `onnx.load` crashes with `SIGSYS` when reading the external-data file on this Android
  build, but `onnx.load(..., load_external_data=False)` and `tract-onnx` both read the
  model correctly.
- Remaining: repeat parity with a fully trained checkpoint and keep the check in the
  release process.

## Next steps

1. Train A0/A1 baselines and SA-LSNet on the CUDA laptop with the documented seeds.
2. Export the best checkpoint with `--dynamo` and rerun the PyTorch/tract parity check.
3. Build the Tauri desktop app on Windows; keep Android build for later.
4. Run the A0-A7 ablation matrix and write measured results into the docs.
5. Cross-region evaluation on Bijie/UAV data if labelled target data is available.

## Planned, not started

- Cross-region evaluation on Bijie/UAV data.
- Android build instructions and CI (cannot build APK in this environment).
- Licence file (needs a decision).
- Foundation-model encoder experiment (future work).

## Known issues and risks

- Full training on this CPU is not practical; use the Windows CUDA laptop.
- ONNX export requires `--dynamo` on this PyTorch 2.14 build; the legacy exporter fails
  before serialization. The dynamo path needs `onnxscript` and writes opset 18.
- `onnx.load` with external data crashes on this Android/Termux build (`SIGSYS`);
  `tract-onnx` and `onnx.load(..., load_external_data=False)` work.
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
