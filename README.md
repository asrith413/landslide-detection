# RGB-Only Structure-Aware Landslide Segmentation

A research-backed, RGB-only landslide segmentation system with a reproducible PyTorch
pipeline and a local-first Tauri 2 application for Windows and Android.

> **Project status:** active development (5th-semester B.Tech project).
> The repository separates *implemented and tested* components from *planned*
> components. See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the live status and
> [`docs/EXPERIMENTAL_PLAN.md`](docs/EXPERIMENTAL_PLAN.md) for the protocol.

## Why RGB-only?

Most strong Landslide4Sense entries use 14 channels, including a DEM and slope layer.
Those layers are often unavailable for fresh UAV imagery or for rapid post-disaster
response. This project asks a stricter question:

> How far can an **RGB-only** model get by learning structural, boundary, and
> object-level cues instead of relying on elevation?

## Proposed contribution: SA-LSNet

**SA-LSNet (Structure-Aware Landslide Segmentation Network)** is an RGB-only model with:

1. **Ten-channel structural prior branch** — luminance, Sobel magnitude, gradient
   orientation (cos 2θ, sin 2θ), Laplacian, local contrast, local entropy,
   morphological top-hat, black-hat, and HSV saturation. The features are
   differentiable, computed on-device, and require no DEM/GIS data.
2. **Shared pretrained RegNet/MobileNet encoder** — multi-scale RGB features from an
   ImageNet-pretrained backbone (configurable; a tiny from-scratch encoder is used
   for tests and fully offline runs).
3. **Gated spatial-channel fusion** — CBAM-style attention plus a learned per-stage
   gate that injects structural evidence only where it helps.
4. **Mask-constrained object-level contrastive learning** — mask-pooled foreground and
   background prototypes, pixel-to-prototype InfoNCE, and prototype separation. This
   is an object-level objective without requiring object proposals.
5. **Lightweight decoder + tiled inference** — exact-size skip fusion, boundary-aware
   loss, and overlap-blended tiling for large UAV/satellite images.

The architecture is deliberately modular so every stage can be ablated.

## Repository layout

```
ml/landslide/          PyTorch package (data, features, models, losses, metrics, training)
ml/configs/            Reproducible experiment configurations
scripts/               Data download/preparation, ablations, export
tests/                 Unit and smoke tests
app/                   Tauri 2 desktop/Android application (TypeScript + Rust)
crates/landslide-core  Pure-Rust ONNX inference, tiling, history, export
docs/                  Literature review, architecture, experimental plan, progress
```

## Quick start (Python)

```bash
# Termux/Linux CPU; use a CUDA-enabled PyTorch wheel on the Windows laptop for training
python -m pip install -r requirements.txt

# 1. Download the official Landslide4Sense release (Zenodo, CC BY 4.0)
python scripts/download_data.py --dataset landslide4sense --out data/raw

# 2. Convert 14-band HDF5 patches to RGB-only NPZ + deterministic splits
python scripts/prepare_landslide4sense.py --raw data/raw --out data/processed/landslide4sense

# 3. Train the RGB baseline and the proposed model
python -m ml.landslide.train --config ml/configs/rgb_baseline.json
python -m ml.landslide.train --config ml/configs/sa_lsnet.json

# 4. Evaluate with IoU, Dice, Precision, Recall, F1, Boundary IoU, latency
python -m ml.landslide.evaluate --checkpoint checkpoints/sa_lsnet/best.pt \
    --data data/processed/landslide4sense --split test

# 4b. Run the ablation matrix and aggregate test metrics
python scripts/run_ablations.py --data data/processed/landslide4sense \
    --out outputs/ablations --seeds 42 43 44

# 5. Export a deployment ONNX model (fixed 128x128 tiles)
python -m ml.landslide.export_onnx --checkpoint checkpoints/sa_lsnet/best.pt \
    --out models/sa_lsnet.onnx
```

## Application

The Tauri app runs inference locally through the pure-Rust `landslide-core` crate
(`tract-onnx`), so Windows and Android do not need a Python runtime or a cloud API.
It supports upload/preview, binary mask + probability map, original-vs-prediction
comparison, adjustable overlay, tiled inference for large images, timing, export,
and local analysis history.

```bash
cd app
npm install
npm run tauri dev       # Windows/Linux desktop
npm run tauri android dev  # requires Android SDK/NDK + Rust Android targets
```

## Datasets and licensing

| Dataset | Role | RGB | Masks | License | Access |
|---|---|---|---|---|---|
| Landslide4Sense (Zenodo) | primary train/val/test | B2/B3/B4 (RGB) | train (val/test labels withheld by organisers) | CC BY 4.0 | https://doi.org/10.5281/zenodo.10463239 |
| Landslide4Sense (HF mirror) | convenience mirror | yes | **mirror annotations are empty** (verified) | MIT/CC BY | https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense |
| Bijie / UAV landslide datasets | cross-region evaluation (planned) | varies | varies | check per dataset | see literature review |

We do **not** commit datasets or model weights. All reported numbers must come from
scripts in this repository and be stored under `outputs/` with the exact config.

## Honest evaluation policy

- No fabricated metrics. Results are only reported after running the included code.
- Official validation/test masks are not available for Landslide4Sense test; we use a
  deterministic, landslide-stratified hold-out from the labelled training release and
  state this clearly. Cross-region claims require a separate labelled dataset.
- Latency is measured on the machine that produced the result and reported with
  hardware, batch size, and input size.

## Citation

If you use this code, cite Landslide4Sense and this repository. See
`docs/LITERATURE_REVIEW.md` for the papers that informed the design.
