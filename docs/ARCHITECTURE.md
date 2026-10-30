# Architecture

Last updated: 2026-10-30

## Goal

SA-LSNet is an **RGB-only** landslide segmentation network. It is designed for
imagery where a DEM, slope layer, NIR, or SWIR bands are unavailable: fresh UAV
flights, phone photos of a slope, or rapid post-event response with RGB tiles.

The model does not claim to beat 14-band models on their own benchmark. It asks how
much of the gap can be recovered from structural and object-level cues that are
computable from RGB alone.

## Data flow

```
RGB tile (H,W,3) uint8
        |
        +--> percentile stretch --> ImageNet normalization --> encoder --> F1..F4
        |                                                          |
        +--> denormalize --> 10-channel structural features --> project --> S1..S4
                                                                   |
                                                     gated fusion per scale
                                                                   |
                                                        EncoderProjector
                                                                   |
                                              decoder (3 upsample blocks + 1 conv block)
                                                       |                    |
                                                embedding head        logits head
                                                (training only)       (B,1,H,W)
```

At inference the deployment wrapper returns only `sigmoid(logits)`.

## 1. Structural prior branch

`ml/landslide/features.py` computes ten deterministic channels from RGB in `[0,1]`.
There are no learned parameters, so the "structural features" ablation is exact.

| # | Channel | Purpose |
|---|---|---|
| 0 | luminance `0.299R + 0.587G + 0.114B` | geometry without colour bias |
| 1 | Sobel magnitude | scarps, edges, debris margins |
| 2 | `cos(2*theta)` orientation | orientation modulo 180 degrees |
| 3 | `sin(2*theta)` orientation | continuous orientation encoding |
| 4 | Laplacian magnitude | fine texture and roughness |
| 5 | local std (7x7) | texture contrast |
| 6 | local Shannon entropy (8 bins, 7x7) | surface heterogeneity |
| 7 | morphological top-hat | bright thin structures |
| 8 | morphological black-hat | dark thin structures |
| 9 | HSV saturation | exposed soil/rock vs vegetation |

All operations use `torch` / `torch.nn.functional` with replicate padding, so the
branch is differentiable, device-agnostic, and exportable. Fixed monotone scalings
keep channels in `[0,1]` without dataset-specific statistics.

## 2. Encoder

`ml/landslide/encoder.py` exposes four scales:

| Backbone | Channels (stride 4/8/16/32) | Notes |
|---|---|---|
| `tiny` | configurable | tests and offline smoke runs |
| `regnet_y_400mf` | 48, 104, 208, 440 | default; 12.49 M total params with full head |
| `regnet_y_800mf` | from torchvision | higher-capacity option |
| `mobilenet_v3_small` | 16, 24, 40, 96 | light option; 3.32 M total params |

ImageNet pretraining is optional. The default configs set `pretrained: true`; use
`--no-pretrained` when the machine is offline or for from-scratch ablations.

## 3. Gated structural fusion

For encoder feature `F_i` and projected structural feature `S_i`:

- `none`: `conv1x1([F_i ; S_i])`
- `gate` (default): CBAM on the concatenation, then a learned per-pixel gate
  `g = sigmoid(conv([F_i ; S_i]))`, and `F_i + g * S_i`
- `cross`: gate at every scale, plus cross-attention between fused deep features and
  structural tokens at the deepest scale

The gate lets the network suppress structural evidence on textured but non-landslide
terrain, which is the main failure mode of handcrafted edge features.

## 4. Mask-constrained object-level contrastive learning

`MaskedObjectContrastiveLoss` in `ml/landslide/losses.py`:

1. Mask-pool normalized embeddings into a foreground prototype and a background
   prototype per image.
2. Classify every pixel against the two prototypes with temperature-scaled cosine
   logits (InfoNCE-style cross entropy).
3. Add a prototype-separation term to prevent collapse on positive-poor batches.
4. Skip images that do not contain at least 8 pixels of both classes.

This is an object-level objective without object proposals, and it uses only the
segmentation mask already present in training data. At inference the head is dropped,
so latency is unchanged.

## 5. Decoder

`EncoderProjector` maps all four encoder scales to a common width. Three `UpBlock`
modules fuse deep-to-shallow features, followed by two 3x3 conv-BN-ReLU blocks and a
1x1 logit head. The final logits are bilinearly resized to the input size.

## 6. Loss

```
L = w_bce * BCEWithLogits(pos_weight) + w_dice * Dice + w_boundary * weighted_BCE
    + w_contrastive * L_object_contrastive
```

Defaults: `w_bce=0.5`, `w_dice=0.5`, `w_contrastive=0.1`, `pos_weight=2.0`,
`w_boundary=0.0` (enabled in the boundary-ablation config).

## 7. Tiled inference

`ml/landslide/tiling.py`:

- `tile_positions` covers the image with 128x128 tiles and 32 px overlap, with a final
  flush tile at the right/bottom edge.
- `stitch_probs` accumulates probability maps with a separable linear ramp and divides
  by the accumulated weight, so overlaps blend smoothly and constant maps are preserved.
- Edge tiles smaller than `2*overlap` clamp their ramp length, avoiding double-assigned
  weights.
- `extract_replicated` pads edge tiles by repeating the last row/column, so the model
  always receives a full tile without black borders.
- Percentile stretch is applied once to the full image before tiling, matching the
  training preprocessing; the Rust core follows the same order.

## 8. Deployment path

Training is PyTorch. Deployment is **pure Rust** through `crates/landslide-core`
(`tract-onnx`), not ONNX Runtime. ONNX Runtime's Android target requires building ORT
from source, which is not practical in this environment. The exported graph is a
fixed-shape 128x128 tile; the Rust core performs tiling, blending, thresholding,
overlay generation, timing, and history storage. `scripts/check_parity.py` compares
PyTorch and Rust outputs on the same image.

## 9. What is deliberately excluded

- DEM, slope, NIR, SWIR, SAR, and multispectral bands.
- Test-time training and cloud APIs.
- Any dependency on GIS libraries at inference time.
- Claims of cross-region generalization without a labelled target dataset.

## 10. Known limitations

- RGB-only performance is materially lower than 14-band performance; see
  `docs/LITERATURE_REVIEW.md`.
- The official Landslide4Sense test masks are withheld. Internal evaluation uses a
  deterministic grouped hold-out from the 4,044 labelled train+validation patches.
- Near-duplicate grouping uses a 64-bit dHash, which is a leakage reducer, not a
  substitute for geographic separation.
- Contrastive loss requires masks with both classes; all-background tiles contribute
  only segmentation loss.
