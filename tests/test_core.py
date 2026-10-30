import unittest

import numpy as np
import torch

from ml.landslide.features import StructuralFeatureExtractor
from ml.landslide.losses import BCEDiceLoss, MaskedObjectContrastiveLoss
from ml.landslide.tiling import pad_to_tile, stitch_probs, tile_positions, unpad


class FeatureTests(unittest.TestCase):
    def test_shape_and_range(self):
        x = torch.rand(2, 3, 64, 64)
        f = StructuralFeatureExtractor()(x)
        self.assertEqual(tuple(f.shape), (2, 10, 64, 64))
        self.assertGreaterEqual(float(f.min()), 0.0)
        self.assertLessEqual(float(f.max()), 1.0)
        self.assertTrue(torch.isfinite(f).all())

    def test_constant_image_has_no_gradient(self):
        x = torch.full((1, 3, 32, 32), 0.5)
        f = StructuralFeatureExtractor()(x)
        self.assertLess(float(f[:, 1].max()), 0.01)


class LossTests(unittest.TestCase):
    def test_bce_dice_is_finite(self):
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = (torch.rand(2, 1, 32, 32) > 0.7).float()
        loss = BCEDiceLoss(boundary_weight=0.1)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_contrastive_skips_empty_masks(self):
        emb = torch.randn(2, 16, 16, 16, requires_grad=True)
        mask = torch.zeros(2, 1, 16, 16)
        loss = MaskedObjectContrastiveLoss()(emb, mask)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_contrastive_is_finite(self):
        emb = torch.randn(2, 16, 16, 16, requires_grad=True)
        mask = torch.zeros(2, 1, 16, 16)
        mask[0, :, :8] = 1
        mask[1, :, 8:] = 1
        loss = MaskedObjectContrastiveLoss()(emb, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class TilingTests(unittest.TestCase):
    def test_cover_and_stride(self):
        tiles = tile_positions(300, 200, tile=128, overlap=32)
        self.assertGreater(len(tiles), 1)
        self.assertEqual(tiles[-1].y + tiles[-1].h, 300)
        self.assertEqual(tiles[-1].x + tiles[-1].w, 200)

    def test_constant_stitch(self):
        h, w, t, ov = 256, 192, 128, 32
        tiles = [(tt, np.full((tt.h, tt.w), 0.7, dtype=np.float32)) for tt in tile_positions(h, w, t, ov)]
        out = stitch_probs(tiles, h, w, ov)
        self.assertEqual(out.shape, (h, w))
        np.testing.assert_allclose(out, 0.7, atol=1e-5)

    def test_small_image_stitch(self):
        h, w, t, ov = 40, 24, 128, 32
        tiles = [(tt, np.full((tt.h, tt.w), 0.4, dtype=np.float32)) for tt in tile_positions(h, w, t, ov)]
        out = stitch_probs(tiles, h, w, ov)
        self.assertEqual(out.shape, (h, w))
        np.testing.assert_allclose(out, 0.4, atol=1e-5)

    def test_pad_unpad(self):
        img = np.zeros((100, 70, 3), dtype=np.uint8)
        padded, orig = pad_to_tile(img, 128)
        self.assertEqual(padded.shape, (128, 128, 3))
        self.assertEqual(unpad(padded, orig).shape, (100, 70, 3))


if __name__ == "__main__":
    unittest.main()
