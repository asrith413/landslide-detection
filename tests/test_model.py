import unittest

import torch

from ml.landslide.encoder import build_encoder
from ml.landslide.losses import BCEDiceLoss, MaskedObjectContrastiveLoss, TotalLoss
from ml.landslide.metrics import SegmentationMetrics, boundary_iou
from ml.landslide.model import DeploymentModel, StructureAwareLandslideNet


class ModelTests(unittest.TestCase):
    def test_full_model_forward_backward(self):
        model = StructureAwareLandslideNet(
            encoder="tiny", pretrained=False, use_structural=True,
            fusion="gate", use_contrastive=True, width=32, decoder_channels=24, embed_dim=16,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(tuple(out["logits"].shape), (2, 1, 128, 128))
        self.assertEqual(tuple(out["embedding"].shape), (2, 16, 32, 32))
        target = (torch.rand(2, 1, 128, 128) > 0.8).float()
        loss = TotalLoss(BCEDiceLoss(), MaskedObjectContrastiveLoss())(out, target)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_deployment_wrapper_returns_probability(self):
        model = StructureAwareLandslideNet(
            encoder="tiny", use_structural=False, use_contrastive=False,
            width=32, decoder_channels=24,
        )
        deploy = DeploymentModel(model).eval()
        with torch.no_grad():
            out = deploy(torch.randn(1, 3, 128, 128))
        self.assertEqual(tuple(out.shape), (1, 1, 128, 128))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rgb_baseline(self):
        model = StructureAwareLandslideNet(
            encoder="tiny", use_structural=False, use_contrastive=False,
            width=32, decoder_channels=24,
        )
        out = model(torch.randn(1, 3, 128, 128))
        self.assertIsNone(out["embedding"])
        self.assertEqual(tuple(out["logits"].shape), (1, 1, 128, 128))


class EncoderTests(unittest.TestCase):
    def test_tiny_has_four_distinct_scales(self):
        feats = build_encoder("tiny")(torch.randn(1, 3, 128, 128))
        self.assertEqual([f.shape[-1] for f in feats], [32, 16, 8, 4])


class MetricTests(unittest.TestCase):
    def test_perfect_and_wrong(self):
        m = SegmentationMetrics()
        logits = torch.full((1, 1, 8, 8), 10.0)
        target = torch.ones(1, 1, 8, 8)
        m.update(logits, target, compute_boundary=False)
        r = m.compute()
        self.assertAlmostEqual(r["iou"], 1.0, places=5)
        m2 = SegmentationMetrics()
        m2.update(torch.full((1, 1, 8, 8), -10.0), target, compute_boundary=False)
        self.assertAlmostEqual(m2.compute()["iou"], 0.0, places=5)

    def test_boundary_iou(self):
        a = (torch.rand(32, 32) > 0.5).numpy().astype("uint8")
        self.assertAlmostEqual(boundary_iou(a, a), 1.0)

    def test_accumulator_boundary_with_channel_dim(self):
        m = SegmentationMetrics()
        logits = torch.full((2, 1, 32, 32), 10.0)
        target = torch.ones(2, 1, 32, 32)
        m.update(logits, target, compute_boundary=True)
        r = m.compute()
        self.assertAlmostEqual(r["boundary_iou"], 1.0)
        self.assertEqual(r["pixels"], 2 * 32 * 32)


if __name__ == "__main__":
    unittest.main()
