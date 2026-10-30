import unittest

import torch

from ml.landslide.losses import BCEDiceLoss, MaskedObjectContrastiveLoss, TotalLoss
from ml.landslide.metrics import SegmentationMetrics, boundary_iou
from ml.landslide.model import StructureAwareLandslideNet


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

    def test_rgb_baseline(self):
        model = StructureAwareLandslideNet(
            encoder="tiny", use_structural=False, use_contrastive=False,
            width=32, decoder_channels=24,
        )
        out = model(torch.randn(1, 3, 128, 128))
        self.assertIsNone(out["embedding"])
        self.assertEqual(tuple(out["logits"].shape), (1, 1, 128, 128))


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


if __name__ == "__main__":
    unittest.main()
