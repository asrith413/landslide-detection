# Experimental plan

Last updated: 2026-10-30

## 1. Research questions

- **RQ1:** How much landslide segmentation quality can an RGB-only model recover on
  Landslide4Sense relative to a matched RGB-only baseline?
- **RQ2:** Do differentiable structural priors improve RGB-only segmentation, and where
  do they help or hurt?
- **RQ3:** Does mask-constrained object-level contrastive learning improve region and
  boundary quality at no inference cost?
- **RQ4:** How do encoder capacity and input resolution trade off against on-device
  latency?
- **RQ5:** Does the model transfer to a different region/sensor without target labels,
  and how much does supervised fine-tuning recover?

## 2. Hypotheses

| ID | Hypothesis | Primary metric |
|---|---|---|
| H1 | Structural features improve F1 and Boundary IoU over an RGB-only CNN baseline | F1, Boundary IoU |
| H2 | Gated fusion beats plain concatenation and no fusion | F1, Boundary IoU |
| H3 | Object-level contrastive loss improves Dice/F1, especially on small landslides | Dice, F1, per-size recall |
| H4 | The light MobileNet variant retains most accuracy at a fraction of latency | F1 vs ms/tile |
| H5 | Cross-region zero-shot performance degrades, and fine-tuning recovers part of it | F1 on target region |

H1-H4 are evaluated on Landslide4Sense. H5 requires a second labelled dataset (Bijie
or a UAV dataset) and is reported separately.

## 3. Data

### Primary: Landslide4Sense (Zenodo, CC BY 4.0)

- 3,799 labelled training patches, 245 validation, 800 test (test masks withheld).
- 14 bands; this project uses **B4/B3/B2 = 0-based indices (3,2,1)** as RGB.
- Preprocessing: per-channel 2nd-98th percentile stretch to uint8, then ImageNet
  normalization. The same stretch is used by the app.
- Splits: deterministic **near-duplicate-grouped** split of the labelled release using
  64-bit dHash and union-find with Hamming threshold 6. Default 70/15/15.
- The official validation split is not used for model selection because its masks are
  withheld in the public release. This is stated in every results table.

Why grouped splits: adjacent patches from the same event can be near-duplicates, and a
random split would overstate performance. Grouping is a leakage reducer, not a
substitute for geographic separation.

### Secondary (planned): Bijie / UAV

- Bijie: 770 landslide and 2,003 non-landslide samples, TripleSat 2018, PNG +
  shapefiles + DEM, from https://gpcv.whu.edu.cn/data/Bijie_pages.html.
- Used only for cross-region evaluation and qualitative figures.
- Any licence constraints are checked before redistribution; images are not committed.

## 4. Configurations

| Config | Structural | Fusion | Contrastive | Encoder | Purpose |
|---|---|---|---|---|---|
| `rgb_baseline.json` | no | none | no | regnet_y_400mf | H1/H3 baseline |
| `structural.json` | yes | gate | no | regnet_y_400mf | H1/H2 |
| `sa_lsnet.json` | yes | gate | yes | regnet_y_400mf | full model |
| `sa_lsnet_light.json` | yes | gate | yes | mobilenet_v3_small | H4 |

All configs share the same data root, seed, optimizer, schedule, augmentation, and
evaluation code. Only the stated factor changes.

## 5. Ablation matrix

Run each row with at least 3 seeds (42, 43, 44) and report mean +/- std.

| Run | Structural | Fusion | Contrastive | Encoder |
|---|---|---|---|---|
| A0 | no | none | no | regnet_y_400mf |
| A1 | yes | none | no | regnet_y_400mf |
| A2 | yes | gate | no | regnet_y_400mf |
| A3 | yes | cross | no | regnet_y_400mf |
| A4 | yes | gate | yes | regnet_y_400mf |
| A5 | no | none | yes | regnet_y_400mf |
| A6 | yes | gate | yes | mobilenet_v3_small |
| A7 | yes | gate | yes | tiny (sanity/offline only) |

Optional: boundary-loss weight in `{0, 0.1, 0.25}` and contrastive weight in
`{0.05, 0.1, 0.2}` on the full model only.

## 6. Training protocol

- Optimizer: AdamW, lr 3e-4, weight decay 1e-2.
- Schedule: linear warmup for 2 epochs, then cosine decay.
- Epochs: up to 30 with early stopping (patience 8) on validation Dice.
- Batch size: 16 on GPU; reduce to 4-8 on CPU.
- Augmentation: horizontal/vertical flips, 90-degree rotations, mild gain/bias, mild
  Gaussian noise. No cropping that removes small landslides.
- Mixed precision only on CUDA.
- Gradient clipping 1.0.
- Model selection: best validation Dice, never test.
- Checkpoints: `checkpoints/<experiment>/best.pt`, `last.pt`, `history.csv`,
  `summary.json`.

CPU training is for smoke tests only. Full runs should use the Windows laptop's CUDA
GPU. Every results table must state the device.

## 7. Metrics

Implemented in `ml/landslide/metrics.py`:

- Pixel IoU and Dice.
- Precision, Recall, F1 at threshold 0.5.
- Boundary IoU with a configurable dilation band.
- Confusion counts (TP/FP/FN/TN).
- Latency: milliseconds per 128x128 tile, batch size 1, warmup excluded, median of
  repeated runs. Report hardware and runtime.

Additional reporting:

- Per-image F1 distribution, not only the dataset mean.
- Recall on small landslides (foreground area below a stated threshold).
- Qualitative overlays for true positives, false positives, and misses.

## 8. Statistical reporting

- 3 seeds per ablation row.
- Report mean, standard deviation, and the individual seed values.
- Use paired tests across images (e.g. bootstrap over images) rather than claiming
  significance from a single run.
- Do not compare to leaderboard numbers that use 14 bands.

## 9. Compute budget

| Setting | Approx. cost |
|---|---|
| Smoke test (64 images, 1 epoch, CPU) | minutes |
| Full run, regnet, 30 epochs, CUDA | hours (measure and record) |
| Full run, mobilenet, 30 epochs, CUDA | lower than regnet; measure |
| ONNX export + Rust parity check | minutes |
| Android latency | device-dependent; record model |

## 10. Threats to validity

- **Split leakage:** dHash grouping reduces but does not eliminate same-event leakage.
- **Label noise:** Landslide4Sense masks are patch-level annotations from multiple
  regions; boundary ambiguity exists.
- **Class imbalance:** many tiles are all background. Report per-image metrics and do
  not rely on pixel accuracy.
- **Threshold sensitivity:** report F1 at 0.5 and, if time permits, the best threshold
  on validation only.
- **Domain shift:** no cross-region claim without target labels.
- **Hardware:** latency numbers are not comparable across devices.

## 11. Deliverables

- Reproducible configs and scripts in this repository.
- `outputs/<experiment>/` with metrics, history, and environment info.
- A results table generated from those files, never typed by hand.
- ONNX model plus a Rust/PyTorch parity report.
- App screenshots and latency measurements on the target device.
