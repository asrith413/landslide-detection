import importlib.util
import unittest
from unittest.mock import patch

import torch

from ml.landslide.baselines import (
    DepthwiseSeparableStructuralAdapter,
    LRASPPMobileNetV3Baseline,
    SegFormerB0Baseline,
    StructuralSegFormerB0,
)
from ml.landslide.blocks import GatedStructuralFusion
from ml.landslide.losses import (
    HardNegativeRegionContrastiveLoss,
    MaskedObjectContrastiveLoss,
)
from ml.landslide.model import build_model


class ResearchLossTests(unittest.TestCase):
    def test_masked_contrastive_uses_foreground_as_class_one(self):
        mask = torch.zeros(1, 1, 4, 8)
        mask[:, :, :, 4:] = 1
        embedding = torch.zeros(1, 2, 4, 8)
        embedding[:, 0] = torch.where(mask[:, 0] > 0, 1.0, -1.0)

        loss = MaskedObjectContrastiveLoss(temperature=0.1, separation_weight=0.0)(embedding, mask)

        self.assertLess(float(loss), 1e-4)

    def test_hard_negative_regions_are_finite_and_backpropagate(self):
        embedding = torch.randn(1, 8, 8, 8, requires_grad=True)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 1:4, 1:4] = 1
        logits = torch.full((1, 1, 8, 8), -8.0, requires_grad=True)
        with torch.no_grad():
            logits[:, :, 1:3, 5:7] = 8.0
            logits[:, :, 5:7, 5:7] = 6.0

        loss = HardNegativeRegionContrastiveLoss(max_negative_regions=2)(
            embedding, mask, segmentation_logits=logits
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(embedding.grad)
        self.assertGreater(float(embedding.grad.abs().sum()), 0.0)
        self.assertIsNone(logits.grad)

    def test_hard_negative_regions_skip_invalid_batches(self):
        embedding = torch.randn(2, 4, 8, 8, requires_grad=True)
        mask = torch.zeros(2, 1, 8, 8)
        logits = torch.full((2, 1, 8, 8), 8.0)

        loss = HardNegativeRegionContrastiveLoss()(embedding, mask, logits)

        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(embedding.grad.abs().sum()), 0.0)


class ResearchModelTests(unittest.TestCase):
    def test_build_model_dispatch_and_pretrained_override(self):
        cases = [
            ("segformer_b0", "SegFormerB0Baseline"),
            ("lraspp_mobilenet_v3", "LRASPPMobileNetV3Baseline"),
            ("csg_segformer_b0", "StructuralSegFormerB0"),
        ]
        for architecture, class_name in cases:
            with self.subTest(architecture=architecture):
                target = f"ml.landslide.model.{class_name}"
                with patch(target) as constructor:
                    expected = constructor.return_value
                    actual = build_model(
                        {"model": {"architecture": architecture, "pretrained": True}},
                        pretrained_override=False,
                    )
                self.assertIs(actual, expected)
                kwargs = constructor.call_args.kwargs
                key = "pretrained_backbone" if architecture.startswith("lraspp") else "pretrained"
                self.assertFalse(kwargs[key])

    def test_structural_adapter_and_gate_shapes(self):
        adapter = DepthwiseSeparableStructuralAdapter(10, 32).eval()
        fusion = GatedStructuralFusion(32, 32, gate_bias=-2.0).eval()
        structural = adapter(torch.randn(2, 10, 16, 20))
        fused = fusion(torch.randn(2, 32, 16, 20), structural)

        self.assertEqual(tuple(structural.shape), (2, 32, 16, 20))
        self.assertEqual(tuple(fused.shape), (2, 32, 16, 20))
        torch.testing.assert_close(
            fusion.gate[-2].bias,
            torch.full_like(fusion.gate[-2].bias, -2.0),
        )

    def test_lraspp_contract_when_torchvision_is_available(self):
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("torchvision is not installed")
        try:
            model = LRASPPMobileNetV3Baseline(pretrained_backbone=False).eval()
        except (ImportError, OSError, RuntimeError) as exc:
            self.skipTest(f"torchvision segmentation is unavailable: {exc}")

        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output["logits"].shape), (1, 1, 64, 64))
        self.assertIsNone(output["embedding"])

    def test_random_segformer_contract_when_transformers_is_available(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers is not installed")
        try:
            model = SegFormerB0Baseline(pretrained=False).eval()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"SegFormer is unavailable: {exc}")

        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output["logits"].shape), (1, 1, 32, 32))
        self.assertIsNone(output["embedding"])

    def test_structural_segformer_contract_when_transformers_is_available(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers is not installed")
        try:
            model = StructuralSegFormerB0(
                pretrained=False, use_contrastive=True, embed_dim=16
            ).eval()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"SegFormer is unavailable: {exc}")

        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output["logits"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(output["embedding"].shape), (1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
