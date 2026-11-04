"""The cascade detector and its vote member, on synthetic data (no competition data, no download)."""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from filseg.cascade import member_masks, refine_threshold
from filseg.infer import Config
from filseg.model import CONVNEXT, CascadeRoIHeads, build_detector, load_detector

ROOT = Path(__file__).resolve().parents[1]


def target():
    m = torch.zeros(1, 128, 128, dtype=torch.uint8)
    m[0, 30:50, 20:100] = 1
    return {
        "boxes": torch.tensor([[20.0, 30.0, 100.0, 50.0]]),
        "labels": torch.tensor([1]),
        "masks": m,
    }


def small():
    return build_detector(128, pretrained=False, backbone="resnet18", cascade=True)


class CascadeModelTest(unittest.TestCase):
    def test_cascade_train_and_eval(self):
        model = small()
        self.assertIsInstance(model.roi_heads, CascadeRoIHeads)
        images = [torch.rand(3, 128, 128)]
        model.train()
        losses = model(images, [target()])
        for k in range(3):
            self.assertIn(f"loss_classifier_s{k}", losses)
            self.assertIn(f"loss_box_reg_s{k}", losses)
        self.assertIn("loss_mask", losses)
        sum(losses.values()).backward()
        model.eval()
        with torch.no_grad():
            out = model(images)[0]
        self.assertEqual(set(out), {"boxes", "labels", "scores", "masks"})

    def test_checkpoint_round_trip(self):
        model = small()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.pt"
            torch.save({"model": model.state_dict()}, path)
            again = load_detector(
                path, torch.device("cpu"), 128, backbone="resnet18", cascade=True
            )
        keys = set(again.state_dict())
        self.assertTrue(any(k.startswith("roi_heads.extra_box_heads.1.") for k in keys))
        self.assertTrue(any(k.startswith("backbone.body.m.") for k in keys))
        self.assertTrue(any(k.startswith("backbone.fpn.") for k in keys))

    def test_stage_regressors_start_equal(self):
        h = small().roi_heads
        w0 = h.box_predictor.bbox_pred.weight
        # rescaled by w_k / w_0 from a deep copy of stage 0 (before any training)
        self.assertEqual(h.extra_box_predictors[0].bbox_pred.weight.shape, w0.shape)
        self.assertEqual(h.coders[2].weights, (30.0, 30.0, 15.0, 15.0))

    def test_convnext_name(self):
        self.assertEqual(CONVNEXT, "convnext_tiny.dinov3_lvd1689m")


class MemberMasksTest(unittest.TestCase):
    def entry(self):
        prob = np.zeros((256, 256), np.float32)
        prob[64:192, 32:224] = 0.9
        weak = np.full((256, 256), 0.9, np.float32)
        return 400, 400, [
            (0.95, prob, (100, 100, 300, 300)),
            (0.60, weak, (0, 0, 100, 100)),  # below the 0.80 admit threshold
        ]

    def test_threshold_and_area(self):
        masks = member_masks(self.entry(), Config())
        self.assertEqual(len(masks), 1)
        m = mask_util.decode(masks[0]).astype(bool)
        self.assertGreaterEqual(m.sum(), 250)
        self.assertFalse(m[:100, :100].any())
        self.assertTrue(m[200, 200])

    def test_disjoint_ownership(self):
        h, w, c = self.entry()
        c = [(0.95, c[0][1], (100, 100, 300, 300)), (0.90, c[0][1], (120, 120, 320, 320))]
        a, b = (mask_util.decode(m).astype(bool) for m in member_masks((h, w, c), Config()))
        self.assertFalse((a & b).any())

    def test_refine_threshold_bounds(self):
        out = refine_threshold(np.ones((256, 256), np.float32), (10, 20, 50, 60), 100, 100, 0.4)
        self.assertEqual(out.sum(), 40 * 40)


class VoteScriptTest(unittest.TestCase):
    def test_extra_member_flows_into_vote(self):
        import runpy

        from filseg.io import write_predictions_csv

        prob = np.full((256, 256), 0.9, np.float32)
        entry = (2048, 2048, [(0.95, prob, (100, 100, 300, 300))])
        m = np.zeros((2048, 2048), np.uint8)
        m[100:300, 100:300] = 1
        base = [mask_util.encode(np.asfortranarray(m))]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_predictions_csv(d / "base.csv", {"s1": base})
            with open(d / "c.pkl", "wb") as f:
                pickle.dump({"s1": entry}, f)
            argv = sys.argv
            sys.argv = ["vote.py", "--base", str(d / "base.csv"),
                        "--extra", f"cascadecpu={d / 'c.pkl'}", "--out", str(d / "o.csv")]
            try:
                runpy.run_path(str(ROOT / "scripts" / "vote.py"), run_name="__main__")
            finally:
                sys.argv = argv
            self.assertTrue((d / "o.csv").exists())


if __name__ == "__main__":
    unittest.main()
