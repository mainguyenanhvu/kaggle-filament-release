"""Synthetic checks of the inference post-processing; needs no data or weights.  python -m unittest discover tests"""

import unittest

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from filseg.infer import (
    Config,
    assign_pixels,
    drop_small,
    nms,
    refine,
    shift_image,
    shift_mask_back,
    square_crop,
    submission_rows,
)
from filseg.model import CropUNetSeedbox, build_detector


def box(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestPostprocess(unittest.TestCase):
    def test_shift_round_trip_drops_padding(self):
        m = box(10, 20, 10, 20) | box(60, 64, 60, 64)
        shifted = (
            np.array(shift_image(Image.fromarray(m.astype(np.uint8) * 255), 16, 16)) > 0
        )
        back = shift_mask_back(shifted, 16, 16)
        self.assertTrue(
            (back == box(10, 20, 10, 20)).all()
        )  # the corner block left the frame

    def test_nms_keeps_highest_of_duplicates(self):
        masks = np.stack([box(0, 10, 0, 10), box(0, 10, 0, 9), box(30, 40, 30, 40)])
        scores, kept = nms(np.array([0.5, 0.9, 0.7]), masks, 0.1)
        self.assertEqual(scores.tolist(), [0.9, 0.7])
        self.assertTrue((kept[0] == masks[1]).all())

    def test_assign_pixels_is_disjoint_and_ordered(self):
        masks = np.stack([box(0, 10, 0, 10), box(5, 15, 5, 15), box(40, 50, 40, 50)])
        out = assign_pixels(np.array([0.85, 0.95, 0.5]), masks, 0.8)
        self.assertEqual(len(out), 2)  # the 0.5 detection is below threshold
        self.assertTrue((out[0] == masks[1]).all())
        self.assertFalse((out[0] & out[1]).any())

    def test_assign_pixels_keeps_best_when_none_pass(self):
        masks = np.stack([box(0, 10, 0, 10), box(30, 40, 30, 40)])
        out = assign_pixels(np.array([0.3, 0.6]), masks, 0.8)
        self.assertEqual(len(out), 1)
        self.assertTrue((out[0] == masks[1]).all())

    def test_drop_small_keeps_largest_if_all_small(self):
        small, smaller = box(0, 5, 0, 5), box(10, 12, 10, 12)
        self.assertEqual(len(drop_small([small, smaller], 250)), 1)
        self.assertTrue((drop_small([small, smaller], 250)[0] == small).all())

    def test_square_crop_stays_inside_image(self):
        for bbox in [(0, 0, 10, 10), (2040, 2040, 2048, 2048), (500, 900, 700, 960)]:
            x0, y0, x1, y1 = square_crop(bbox, 2048, 2048, 1.8, 96)
            self.assertEqual(x1 - x0, y1 - y0)
            self.assertGreaterEqual(x1 - x0, 96)
            self.assertTrue(0 <= x0 <= bbox[0] and bbox[2] <= x1 <= 2048)

    def test_submission_rows_fallback_and_rle(self):
        rows = submission_rows("img", [], (64, 64))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].startswith("img_1,"))
        m = box(3, 9, 4, 20)
        stem_n, counts = submission_rows("img", [m], (64, 64))[0].split(",", 1)
        decoded = mask_util.decode({"size": [64, 64], "counts": counts.encode()})
        self.assertEqual(stem_n, "img_1")
        self.assertTrue((decoded.astype(bool) == m).all())


class TestModels(unittest.TestCase):
    def test_refiner_shapes_and_fallback(self):
        torch.manual_seed(0)
        refiner = CropUNetSeedbox(pretrained=False).eval()
        gray = np.random.default_rng(0).integers(0, 255, (300, 300), dtype=np.uint8)
        mask = box(100, 120, 100, 180, size=300)
        out = refine(
            refiner, gray, gray, mask, Config(refine_thresh=1.01), torch.device("cpu")
        )
        self.assertTrue(
            (out == mask).all()
        )  # nothing reaches p >= 1.01, so the detector mask stays

    def test_detector_state_dict_matches_training_build(self):
        # inference builds without pretrained weights; the parameter set must be the same
        bare = build_detector(pretrained=False).state_dict()
        self.assertIn("roi_heads.mask_predictor.mask_fcn_logits.weight", bare)
        self.assertEqual(bare["roi_heads.box_predictor.cls_score.weight"].shape[0], 2)
        self.assertEqual(bare["rpn.head.cls_logits.weight"].shape[0], 3)
        five = build_detector(
            anchor_ratios=(0.25, 0.5, 1, 2, 4), pretrained=False
        ).state_dict()
        self.assertEqual(five["rpn.head.cls_logits.weight"].shape[0], 5)


if __name__ == "__main__":
    unittest.main()
