"""Synthetic checks of the rubric metrics; needs no competition data.  python -m unittest discover tests"""

import unittest

import numpy as np
from pycocotools import mask as mask_util

from filseg.metrics import bootstrap, evaluate


def rle(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), dtype=np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(m)


class TestMetrics(unittest.TestCase):
    def test_perfect_match(self):
        a = rle(0, 10, 0, 10)
        ev = evaluate({"s": [[a]]}, {"s": [a]})
        self.assertEqual((ev.tp, ev.fp, ev.fn), (1, 0, 0))
        self.assertAlmostEqual(ev.pq, 1.0)
        self.assertAlmostEqual(float(ev.pair_dice[0]), 1.0)

    def test_every_record_counts(self):
        # two annotators; only the first drew the filament -> TP for record 1, FP for record 2
        a = rle(0, 10, 0, 10)
        ev = evaluate({"s": [[a], []]}, {"s": [a]})
        self.assertEqual((ev.tp, ev.fp, ev.fn, ev.n_records), (1, 1, 0, 2))
        self.assertAlmostEqual(ev.pq, 1 / 1.5)

    def test_dice_from_iou_and_split(self):
        # GT 10x20 split into two 10x10 predictions: each IoU 0.5 (not a hit) -> 1 FN, 2 FP, 1:n degree 2
        gt = rle(0, 10, 0, 20)
        p1, p2 = rle(0, 10, 0, 10), rle(0, 10, 10, 20)
        ev = evaluate({"s": [[gt]]}, {"s": [p1, p2]})
        self.assertEqual((ev.tp, ev.fp, ev.fn), (0, 2, 1))
        self.assertEqual(ev.gt_degree[2], 1)
        self.assertEqual(ev.pred_degree[1], 2)
        np.testing.assert_allclose(ev.pair_dice, [2 / 3, 2 / 3])

    def test_merge_is_many_to_one(self):
        g1, g2 = rle(0, 10, 0, 10), rle(0, 10, 12, 22)
        ev = evaluate({"s": [[g1, g2]]}, {"s": [rle(0, 10, 0, 22)]})
        self.assertEqual(ev.pred_degree[2], 1)
        self.assertEqual(ev.summary()["many_to_one"]["multi"], 1)

    def test_missing_predictions_are_false_negatives(self):
        ev = evaluate({"s": [[rle(0, 5, 0, 5)]]}, {})
        self.assertEqual((ev.tp, ev.fp, ev.fn), (0, 0, 1))
        self.assertEqual(ev.pq, 0.0)

    def test_hit_degree_counts_only_matches(self):
        # the 10x20 GT split in two halves overlaps both (degree 2 at IoU > 0) but matches neither
        gt = rle(0, 10, 0, 20)
        ev = evaluate({"s": [[gt]]}, {"s": [rle(0, 10, 0, 10), rle(0, 10, 10, 20)]})
        self.assertEqual(ev.gt_degree[2], 1)
        self.assertEqual(ev.gt_hit_degree[0], 1)
        # two overlapping GT records drawn by one annotator, one prediction matching both
        g1, g2 = rle(0, 10, 0, 10), rle(0, 10, 0, 11)
        ev = evaluate({"s": [[g1, g2]]}, {"s": [g1]})
        self.assertEqual(ev.pred_hit_degree[2], 1)
        self.assertEqual(ev.summary()["many_to_one_iou>0.5"]["multi"], 1)
        self.assertEqual(ev.tp, 2)

    def test_tp_distributions_cover_matched_pairs_only(self):
        a, b = rle(0, 10, 0, 10), rle(0, 10, 0, 15)  # IoU 2/3, a match
        far = rle(30, 40, 30, 40)
        ev = evaluate({"s": [[b, far]]}, {"s": [a, rle(35, 45, 35, 45)]})
        s = ev.summary()
        self.assertEqual(s["tp_iou"]["n"], 1)
        self.assertAlmostEqual(s["tp_iou"]["mean"], round(2 / 3, 4))
        self.assertAlmostEqual(s["tp_dice"]["mean"], round(0.8, 4))
        self.assertEqual(
            s["iou"]["n"], 2
        )  # the overlapping non-match is in the pair distribution

    def test_bootstrap_interval_brackets_point_estimate(self):
        a = rle(0, 10, 0, 10)
        records = {f"s{i}": [[a]] for i in range(6)} | {"t": [[a], []]}
        preds = {k: [a] for k in records}
        ci = bootstrap(records, preds, n_boot=500, seed=1)
        pq = evaluate(records, preds).pq
        self.assertLessEqual(ci["pq"][0], pq)
        self.assertGreaterEqual(ci["pq"][1], pq)
        self.assertEqual(ci["sq"], [1.0, 1.0])
        self.assertEqual(bootstrap(records, preds, n_boot=500, seed=1), ci)


if __name__ == "__main__":
    unittest.main()
