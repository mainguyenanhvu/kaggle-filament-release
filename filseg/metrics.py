"""Challenge metrics, scored over every annotator record.

The organisers' self-evaluation notebook compares each (annotator, image) record against the
prediction set of that image and pools the counts over all records. An image drawn by three
annotators therefore counts three times, and a prediction is a false positive once per annotator
who did not draw it. This module reimplements those rules on run-length-encoded masks:

* Panoptic Quality: a (ground truth, prediction) pair is a true positive when IoU > 0.5, with no
  one-to-one assignment; a prediction that hits no ground truth is a false positive, and a ground
  truth that no prediction hits is a false negative. PQ = sum(TP IoU) / (TP + FP/2 + FN/2).
* IoU and Dice distributions: every pair with overlap > 0, pooled over all records.
* One-to-many (1:n): for each ground-truth filament, the number of predictions it overlaps.
  Many-to-one (n:1): for each prediction, the number of ground-truth filaments it overlaps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

HIT_IOU = 0.5


@dataclass
class Evaluation:
    """Pooled counts and pair scores over a set of annotator records."""

    tp_iou: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    n_records: int = 0
    pair_iou: list[float] = field(default_factory=list)
    gt_degree: Counter = field(default_factory=Counter)
    pred_degree: Counter = field(default_factory=Counter)

    @property
    def tp(self) -> int:
        return len(self.tp_iou)

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return sum(self.tp_iou) / denom if denom else 0.0

    @property
    def sq(self) -> float:
        return float(np.mean(self.tp_iou)) if self.tp_iou else 0.0

    @property
    def rq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom else 0.0

    @property
    def pair_dice(self) -> np.ndarray:
        iou = np.asarray(self.pair_iou, dtype=float)
        return (
            2 * iou / (1 + iou)
        )  # Dice = 2J / (1 + J), exact for a pair of binary masks

    def summary(self) -> dict:
        """JSON-ready table of every rubric quantity."""
        return {
            "pq": round(self.pq, 5),
            "sq": round(self.sq, 4),
            "rq": round(self.rq, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_records": self.n_records,
            "iou": _describe(np.asarray(self.pair_iou, dtype=float)),
            "dice": _describe(self.pair_dice),
            "one_to_many": _degrees(self.gt_degree),
            "many_to_one": _degrees(self.pred_degree),
        }


def _describe(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 4),
        "q25": round(float(q25), 4),
        "median": round(float(q50), 4),
        "q75": round(float(q75), 4),
        "frac_gt_0.5": round(float((x > 0.5).mean()), 4),
    }


def _degrees(c: Counter) -> dict:
    """Histogram of overlap degree; 'multi' counts the objects with degree >= 2."""
    hist = {str(k): c[k] for k in sorted(c)}
    return {
        "hist": hist,
        "multi": sum(v for k, v in c.items() if k >= 2),
        "n": sum(c.values()),
    }


def score_record(ev: Evaluation, gt: list[dict], pred: list[dict]) -> None:
    """Add one annotator record (its ground-truth RLEs) against the image's predicted RLEs."""
    ev.n_records += 1
    if not gt:
        ev.fp += len(pred)
        ev.pred_degree.update([0] * len(pred))
        return
    if not pred:
        ev.fn += len(gt)
        ev.gt_degree.update([0] * len(gt))
        return
    iou = np.asarray(mask_util.iou(gt, pred, [0] * len(pred)), dtype=float).reshape(
        len(gt), len(pred)
    )
    hit = iou > HIT_IOU
    ev.tp_iou += iou[hit].tolist()
    ev.fp += int((hit.sum(0) == 0).sum())
    ev.fn += int((hit.sum(1) == 0).sum())
    ev.pair_iou += iou[iou > 0].tolist()
    touch = iou > 0
    ev.gt_degree.update(touch.sum(1).tolist())
    ev.pred_degree.update(touch.sum(0).tolist())


def evaluate(
    records: dict[str, list[list[dict]]], preds: dict[str, list[dict]]
) -> Evaluation:
    """records: image stem -> list of annotator records, each a list of GT RLEs.
    preds: image stem -> list of predicted RLEs (a missing stem means no predictions)."""
    ev = Evaluation()
    for stem, recs in records.items():
        pred = preds.get(stem, [])
        for gt in recs:
            score_record(ev, gt, pred)
    return ev
