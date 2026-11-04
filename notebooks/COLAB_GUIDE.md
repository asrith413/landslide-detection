# Google Colab Training Guide

[Open `Landslide_Training_Colab.ipynb` in Google Colab](https://colab.research.google.com/github/asrith413/landslide-detection/blob/main/notebooks/Landslide_Training_Colab.ipynb)

The notebook uses Colab only as the GPU training host. Source code remains in this repository, while datasets, checkpoints, metrics, plots, and exported models persist in Google Drive.

## Scope

The workflow trains and compares:

- SegFormer-B0 RGB baseline;
- MobileNetV3-Large/LR-ASPP RGB baseline;
- CSG-SegFormer-B0 with the project's structural features, compact adapters, residual gates, and training-only hard-negative region contrast.

It evaluates IoU, Dice, Precision, Recall, F1, and Boundary IoU globally and by held-out CAS event. Optional HR-GLDD evaluation uses the official RGB channels and discards NIR without fine-tuning.

The notebook does not include synthetic scientific training, cached scores, or claimed results. Metrics exist only after a real run completes.

## Prerequisites

1. A Google account with enough Drive capacity for the selected data and run artifacts.
2. Colab with **Runtime > Change runtime type > GPU** selected. A Tesla T4 is suitable; Colab may allocate a different GPU.
3. Access to the CAS Landslide Dataset and acceptance of its CC BY-NC 4.0 and source-specific imagery terms.
4. The complete nine-event CAS collection for a scientific run. A single Wenchuan archive cannot provide event-disjoint train, validation, and test roles.

The official CAS archives total about 9.19 GB compressed. Keep additional space for extraction, checkpoints, ONNX files, and temporary download parts. The notebook reports Drive and GPU memory before doing expensive work.

## Persistent Layout

The default parameter cell uses:

```text
MyDrive/landslide-colab/
  datasets/
    cas/
      archives/
      extracted/
      cas_event_manifest.csv
      manifest-validation-cache.json
    hr-gldd/
  runs/
    cas-event-v1/
      resolved_configs/
      segformer_b0_cas/
      lraspp_mobilenet_v3_cas/
      csg_segformer_b0_cas/
```

Change `DRIVE_PROJECT_ROOT` in the parameter cell if needed. No other code cell requires path edits.

## CAS Access Modes

### Existing Drive Data

Set:

```python
CAS_MODE = "mounted"
```

Then use one of these arrangements:

- Place extracted archives under the directory names declared in `ml/configs/cas_colab.json`. The notebook can build a deterministic manifest.
- Point `CAS_DATA_ROOT` and `CAS_MANIFEST_PATH` at an existing portable manifest. Relative image and mask paths must stay under `CAS_DATA_ROOT`.

The manifest requires these columns:

```csv
image,mask,split,event
```

Additional columns are retained but not required for loading. Paths must not escape the data root. Masks must use binary `0/1`, `0/255`, or `0/65535` values.

### One-Time Zenodo Download

Set:

```python
CAS_MODE = "download"
```

The notebook reads the official URL, byte count, and MD5 for every archive from `ml/configs/cas_colab.json`. Downloads use persistent `.part` files, resume only when the server honors byte ranges, verify size and checksum, and reuse already verified archives. ZIP extraction rejects path traversal and writes a checksum-bound completion marker so reruns do not extract again.

Do not set `REBUILD_CAS_MANIFEST = True` unless the extracted collection changed.

## Event Policy

The committed policy keeps every sensor and resolution from one event family in one role:

| Role | Event families |
|---|---|
| Train | Jiuzhai Valley, Hokkaido Iburi-Tobu, Lombok |
| Validation | Moxi/Luding, Tiburon Peninsula, Wenchuan |
| Test | Longxi River, Palu, Mengdong Township |

Moxi town and Moxitaidi are aliases of the same Moxi/Luding event family. Longxi UAV/SAT and Tiburon Planet/Sentinel are likewise kept together. Validation fails if an event crosses roles, a required family is absent, paths are duplicated, files are missing, dimensions differ, or masks are not binary.

`FULL_DATA_VALIDATION = True` decodes every pair on the first run. A cache keyed by manifest hash, policy hash, data root, and validation options avoids repeatedly decoding the 9.19 GB collection. Manifest structure and file existence are still checked on every run.

## Running

1. Open the notebook with the link at the top of this guide.
2. Select a GPU runtime.
3. Edit the parameter cell for Drive paths and acquisition mode.
4. Run all cells in order.
5. Approve the Google Drive mount when prompted.

The default `EXPERIMENT_PRESETS` list runs all three models. Remove entries to run one model, or use `BATCH_SIZE_OVERRIDES` if GPU memory is limited. Keep the same `RUN_NAME` when continuing an interrupted experiment.

For a final reproducible experiment, set `REPO_REVISION` to the full Git commit SHA recorded by a successful setup cell rather than `main`.

## Checkpoint Resume

After every epoch, training atomically replaces `last.pt` in the experiment's Drive directory. `best.pt` tracks the best validation Dice checkpoint. Checkpoints include:

- model, optimizer, scheduler, and AMP scaler state;
- completed epoch, early-stopping state, and complete history;
- Python, NumPy, PyTorch, CUDA, data-loader, and augmentation RNG state;
- resolved experiment configuration and CAS manifest SHA-256.

Rerunning the training cell resumes from `last.pt` when present. Resume is rejected if the experiment, model configuration, or manifest hash changed. Increase `EPOCHS` to continue a completed shorter run without changing `RUN_NAME`.

## GPU Memory

The notebook uses CUDA mixed precision, gradient accumulation, pinned-memory transfers, gradient clipping, and optional epoch-level CUDA cache release. Initial batch sizes are conservative but cannot fit every Colab GPU allocation.

For an out-of-memory error:

```python
BATCH_SIZE_OVERRIDES = {
    "segformer_b0_cas": 2,
    "csg_segformer_b0_cas": 1,
}
```

Do not change preprocessing, event roles, or test data to solve memory pressure. Lower batch size and retain gradient accumulation instead.

## Held-Out and External Evaluation

CAS test metrics are computed only after training and are written globally and per event. The notebook does not use test metrics for checkpoint selection.

To evaluate HR-GLDD when its official arrays are available:

```python
RUN_HR_GLDD_EVALUATION = True
```

Set `DOWNLOAD_HR_GLDD_TEST = True` to download and MD5-check only `testX.npy` and `testY.npy`, or place those files under `HR_GLDD_ROOT`. The loader memory-maps the arrays, verifies their layout and binary masks, selects RGB, and explicitly discards the fourth NIR channel. It does not fine-tune on HR-GLDD.

## Export and Quantization

For each trained model, the deployment stage:

1. Exports a static `1x3x512x512` ONNX probability model.
2. Runs `onnx.checker`.
3. Compares ONNX Runtime CPU output with PyTorch on the same normalized real CAS tile.
4. Rejects numerical parity failures instead of publishing the model.
5. Attempts ONNX Runtime dynamic INT8 quantization.
6. Reports quantization as skipped when the graph has no supported operators.
7. Benchmarks warmed PyTorch GPU, ONNX CPU, and usable INT8 CPU models with median, mean, minimum, and p95 latency.

Dynamic quantization primarily targets linear operators. LR-ASPP is convolution-heavy and may honestly report that dynamic quantization is unsupported. No quantized accuracy is invented.

## Saved Artifacts

Each experiment directory contains, when applicable:

```text
config.json
runtime.json
best.pt
last.pt
history.json
history.csv
summary.json
training_progress.png
evaluation_cas_test.json
evaluation_hr_gldd_test.json
held_out_prediction.png
model.onnx
model.json
model.int8.onnx
model.int8.json
deployment_report.json
artifact_manifest.json
```

`artifact_manifest.json` records SHA-256 and size for every artifact. Keep the resolved config, Git SHA, manifest validation report, and runtime report with any reported result.

## Troubleshooting

### CUDA is unavailable

Reconnect after selecting a GPU runtime. The notebook deliberately fails rather than silently training on CPU. Check `nvidia-smi`, `torch.version.cuda`, and `torch.cuda.is_available()` in the preflight output.

### Dependency installation breaks CUDA

Restart the GPU runtime and rerun from the top. The notebook checks CUDA before and after installation. Colab's existing CUDA-enabled Torch satisfies the project lower bound and should not be replaced.

### Manifest policy coverage fails

The complete scientific policy requires all nine event families. Check event aliases and verify all 16 official archive directories are present. Do not relabel individual tiles to fill a missing role.

### Validation is slow

The first full decode is intentionally expensive. Keep the validation cache on Drive. Later runs reuse it only when the manifest, policy, root, and validation options match.

### Drive disconnects

Reconnect Drive and rerun setup, validation, configuration, then training. The same `RUN_NAME` resumes from `last.pt`. A disconnect during the first epoch cannot be resumed until that epoch checkpoint exists.

### ONNX export fails

Keep the PyTorch checkpoint and error output. Do not mark export as successful. Confirm `onnx`, `onnxruntime`, and `onnxscript` versions from the setup cell, then test the same checkpoint without changing preprocessing.

## Local Validation

The notebook itself is data- and GPU-dependent, but its structure and project integration can be checked locally:

```bash
python -m compileall -q ml scripts tests
python -m unittest discover -s tests -v
python scripts/validate_colab_notebook.py
```

These checks do not fabricate or substitute for real CAS training results.
