# Literature review

Last updated: 2026-10-30

This review is restricted to claims verified from the cited sources during this
project. Numbers are quoted only when the source reports them. The goal is to justify
the design choices in SA-LSNet and to define an honest comparison protocol.

## 1. The benchmark and why RGB-only is hard

**Landslide4Sense** is the primary benchmark. It contains 3,799 training, 245
validation, and 800 test patches of 128x128 pixels, with 14 layers: Sentinel-2 B1-B12,
a DEM, and slope from ALOS PALSAR. The official test masks are withheld by the
organisers. The labelled release is CC BY 4.0.

The competition outcome paper reports a baseline test F1 of **59.92%** and winning
entries at **74.54%**, **73.99%**, **73.50%**, and **71.29%** F1. Those top results use
the full 14-band stack.

Direct evidence that RGB-only is the hard setting comes from Ghorbanzadeh et al.
(2024), who evaluate seven band-selection scenarios with a supervised U-Net. Their
fully supervised U-Net reaches its **lowest F1 with RGB alone (about 60%)** and its
best with **RGB + NIR + slope (above 74%)**. The same paper shows that self-supervised
pretraining helps most in the low-label regime: with 1% of labels, the SSL model
outperforms the supervised model trained on the same 40 patches by large margins, but
still trails the fully supervised model when all labels are used.

Orynbaikyzy et al. (2025) independently find that optical bands dominate SAR for
landslide mapping, but that **RGB + NIR is the weakest optical combination (F1 0.69)**
while **RGB + NIR + SWIR reaches F1 0.83**. Their result reinforces two points:
spectral range matters, and RGB-only is a deliberately constrained problem.

**Implication for this project:** a credible RGB-only result should be compared to an
RGB-only baseline trained with the same split and preprocessing, not to 14-band
leaderboard numbers. Cross-paper comparisons must state the band setting.

## 2. Structural priors from RGB

Landslides have geometric signatures that are partly visible in RGB: scarps, debris
tails, roughness contrasts, and sharp vegetation boundaries. Handcrafted structural
channels are attractive because they are cheap, interpretable, and available on-device.

Related lines of work include boundary-refinement and edge-aware segmentation
(IDGBR, arXiv:2507.01573) and contrastive boundary learning (arXiv:2203.05272). These
methods show that boundary quality is a distinct objective from region overlap and
that boundary-aware supervision improves delineation.

SA-LSNet differs by injecting a **fixed 10-channel structural prior** (luminance,
Sobel, orientation, Laplacian, local std, entropy, top-hat, black-hat, saturation)
into a pretrained RGB encoder through a learned gate. The gate is important: naive
edge concatenation amplifies texture in non-landslide terrain, a known failure mode in
mountainous scenes.

## 3. Attention and multi-scale fusion

CBAM (Woo et al., arXiv:1807.06521) provides channel and spatial attention at low cost.
U-Net (Ronneberger et al., arXiv:1505.04597) remains a strong segmentation baseline and
motivates exact-size skip fusion. RegNet (Radosavovic et al., arXiv:2003.13678) and
MobileNetV3 (Howard et al., arXiv:1905.02244) provide practical encoder choices for CPU
and mobile deployment. SegFormer (Xie et al., arXiv:2105.15203) is a reference point for
lightweight transformer segmentation but is not used here because the deployment target
is a small ONNX graph executed by `tract`.

## 4. Contrastive and object-level learning

Supervised contrastive learning (Khosla et al., arXiv:2004.11362) shows that pulling
same-class embeddings together and pushing different-class embeddings apart improves
representation quality and robustness. Pixel-to-prototype contrast (Wang et al.,
arXiv:2110.07110) adapts this idea to dense prediction by contrasting pixels with class
prototypes.

SA-LSNet uses a simpler, mask-only variant: foreground and background prototypes are
mask-pooled per image, pixels are classified against both prototypes with a
temperature-scaled cosine objective, and a separation term prevents collapse. This
requires no object proposals, no memory bank, and no extra inference cost. It is an
object-level regularizer, not a claim of a new contrastive framework.

## 5. Domain shift and cross-region generalization

Landslide appearance changes with sensor, season, illumination, vegetation, and
geology. Recent work addresses this with:

- cross-temporal style adaptation and progressive pseudo-label upgrading (PluTsa,
  IEEE TGRS 2024, DOI 10.1109/TGRS.2024.3425863);
- style-aware spatial-frequency spiking convolution for cross-domain mapping (SSFSC,
  DOI 10.1080/15481603.2025.2559457);
- multi-source domain adaptation for susceptibility mapping (MDACNN, ScienceDirect
  S1674987125000581);
- unsupervised change detection with geospatial foundation models (Leonardi et al.,
  DOI 10.1080/17538947.2025.2547292).

The last study is a useful caution: on its GLaD4CD benchmark, a band-thresholding
baseline reaches F1 48.41%, while an SSL4EO DINO-based metric-learning model reaches
31.68% and RGB deep change detection only 19.66%. Foundation-model features are not a
guarantee of better landslide discrimination, especially without task adaptation.

**Implication:** this project treats cross-region evaluation as a separate experiment
requiring labelled target data. It does not claim geographic generalization from
Landslide4Sense alone.

## 6. Foundation models and efficiency

GEO-Bench-2 (arXiv:2511.15658) evaluates geospatial foundation models across many
tasks and finds no single model dominates. EO-specific models tend to do better on
multispectral tasks, while high-resolution vision models can be stronger on fine
detail. For an RGB-only, on-device application, a compact CNN with ImageNet
pretraining is a defensible baseline; a foundation-model encoder is future work, not a
prerequisite.

Deployment constraints are real: ONNX Runtime's Android target generally requires
building ORT from source, which is impractical in a Termux-only workflow. A pure-Rust
`tract-onnx` runtime avoids that toolchain and keeps the app local-first.

## 7. Gap addressed by this project

1. Most strong Landslide4Sense results depend on DEM/slope/multispectral bands. There
   is limited transparent reporting of **RGB-only** performance with matched baselines.
2. Structural priors are often hand-engineered outside the training graph. This project
   makes them differentiable and exportable.
3. Contrastive objectives are usually evaluated on classification or with external
   pretraining. This project tests a mask-only object-level term inside a segmentation
   model and ablates it directly.
4. Deployment is often omitted. This project includes a pure-Rust inference core and a
   Tauri app plan, with latency reported on the actual device.

## 8. Design decisions derived from the literature

| Decision | Evidence |
|---|---|
| Always report an RGB-only baseline | RGB is the weakest band setting in Ghorbanzadeh et al. (2024) and Orynbaikyzy et al. (2025) |
| Use boundary-aware metrics | Boundary refinement is a distinct objective (IDGBR, CBL) |
| Gate structural features | Naive edges amplify non-landslide texture |
| Keep contrastive head training-only | No inference cost; object-level signal is auxiliary |
| Use grouped splits | Patches from the same event can leak across random splits |
| Avoid cross-region claims without target labels | Domain shift is documented and severe |
| Use `tract-onnx` for mobile | ORT Android build is impractical here |

## References

- Landslide4Sense dataset: https://doi.org/10.5281/zenodo.10463239
- Landslide4Sense competition outcome: DOI 10.1109/JSTARS.2022.3220845
- Ghorbanzadeh et al., Contrastive Self-Supervised Learning for Globally Distributed
  Landslide Detection, IEEE Access 2024: https://doi.org/10.1109/ACCESS.2024.3449447
- Orynbaikyzy et al., Landslide mapping with deep learning: the role of pre-/post-event
  SAR features and multi-sensor data fusion, 2025:
  https://doi.org/10.1080/15481603.2025.2502214
- Li et al., PluTsa, IEEE TGRS 2024: https://doi.org/10.1109/TGRS.2024.3425863
- Yu et al., SSFSC, 2025: https://doi.org/10.1080/15481603.2025.2559457
- Leonardi et al., Geospatial foundation models for unsupervised landslide change
  detection, 2025: https://doi.org/10.1080/17538947.2025.2547292
- Su et al., MDACNN, 2025: https://www.sciencedirect.com/science/article/pii/S1674987125000581
- Khosla et al., Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
- Wang et al., Pixel-to-Prototype Contrast: https://arxiv.org/abs/2110.07110
- Woo et al., CBAM: https://arxiv.org/abs/1807.06521
- Radosavovic et al., RegNet: https://arxiv.org/abs/2003.13678
- Howard et al., MobileNetV3: https://arxiv.org/abs/1905.02244
- Ronneberger et al., U-Net: https://arxiv.org/abs/1505.04597
- Xie et al., SegFormer: https://arxiv.org/abs/2105.15203
- IDGBR: https://arxiv.org/abs/2507.01573
- Contrastive Boundary Learning: https://arxiv.org/abs/2203.05272
- GEO-Bench-2: https://arxiv.org/abs/2511.15658
- Bijie landslide dataset: https://gpcv.whu.edu.cn/data/Bijie_pages.html
- Landslide semantic segmentation with vision models (L4S + Bijie, RGB):
  https://arxiv.org/abs/2403.10127
