"""Synthetic checks of the CPR strip refiner (no data, no weights)."""

import unittest

import numpy as np
import torch
from pycocotools import mask as mask_util

from filseg.cpr import StripNet, frame, refine_image


def _rle(b):
    return mask_util.encode(np.asfortranarray(b.astype(np.uint8)))


class TestCPR(unittest.TestCase):
    def setUp(self):
        self.img = np.full((2048, 2048), 128, np.uint8)
        a = np.zeros((2048, 2048), bool)
        a[500:512, 300:700] = True
        b = np.zeros((2048, 2048), bool)
        b[1200:1400, 1000:1010] = True
        self.masks = [a, b]
        self.img[a | b] = 40

    def test_frame_follows_the_spine(self):
        fr = frame(self.masks[0])
        self.assertTrue(16 <= fr["H"] <= 64)
        self.assertGreater(fr["core"][1] - fr["core"][0], 350)

    def test_untrained_net_keeps_every_mask(self):
        # zero-initialised output: the strip logit equals the prior's, so the guard keeps each mask
        net = StripNet().eval()
        out = refine_image(net, self.img, [_rle(m) for m in self.masks], torch.device("cpu"))
        self.assertEqual(len(out), 2)
        for r, m in zip(out, self.masks):
            d = mask_util.decode(r).astype(bool)
            self.assertGreaterEqual((d & m).sum() / (d | m).sum(), 0.9)

    def test_empty_image(self):
        self.assertEqual(refine_image(StripNet().eval(), self.img, [], torch.device("cpu")), [])


if __name__ == "__main__":
    unittest.main()
