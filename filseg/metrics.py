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
  "Overlap" is IoU > 0, as in the notebook's figure; the same histograms are also reported at the
  match threshold (IoU > 0.5), where a degree >= 2 is a split or a merge that the scorer counts.
* Matched-pair distributions: IoU and Dice of the true-positive pairs only.
* Uncertainty: a bootstrap over images (all records of an image move together).
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
    gt_hit_degree: Counter = field(default_factory=Counter)
    pred_hit_degree: Counter = field(default_factory=Counter)

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

    @property
    def tp_dice(self) -> np.ndarray:
        iou = np.asarray(self.tp_iou, dtype=float)
        return 2 * iou / (1 + iou)

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
            "tp_iou": _describe(np.asarray(self.tp_iou, dtype=float)),
            "tp_dice": _describe(self.tp_dice),
            "one_to_many": _degrees(self.gt_degree),
            "many_to_one": _degrees(self.pred_degree),
            "one_to_many_iou>0.5": _degrees(self.gt_hit_degree),
            "many_to_one_iou>0.5": _degrees(self.pred_hit_degree),
        }


def _describe(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 4),
        "q10": round(float(q10), 4),
        "q25": round(float(q25), 4),
        "median": round(float(q50), 4),
        "q75": round(float(q75), 4),
        "q90": round(float(q90), 4),
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
        ev.pred_hit_degree.update([0] * len(pred))
        return
    if not pred:
        ev.fn += len(gt)
        ev.gt_degree.update([0] * len(gt))
        ev.gt_hit_degree.update([0] * len(gt))
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
    ev.gt_hit_degree.update(hit.sum(1).tolist())
    ev.pred_hit_degree.update(hit.sum(0).tolist())


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


def _counts(ev: Evaluation) -> np.ndarray:
    return np.array([sum(ev.tp_iou), ev.tp, ev.fp, ev.fn], dtype=float)


def _pq_sq_rq(c: np.ndarray) -> dict:
    denom = c[1] + 0.5 * c[2] + 0.5 * c[3]
    return {
        "pq": round(float(c[0] / denom), 5) if denom else 0.0,
        "sq": round(float(c[0] / c[1]), 4) if c[1] else 0.0,
        "rq": round(float(c[1] / denom), 4) if denom else 0.0,
        "tp": int(c[1]),
        "fp": int(c[2]),
        "fn": int(c[3]),
    }


def human_baseline(
    records: dict[str, list[list[dict]]],
    preds: dict[str, list[dict]],
    n_boot: int = 0,
    seed: int = 0,
) -> dict:
    """A single annotator scored as a predictor, beside the predictions, on the same references.

    On every image with at least two non-empty annotator records, each record in turn is the
    predictor and the remaining records are the references, pooled as in `evaluate`; the
    predictions are scored against exactly the same reference sets. With n_boot > 0 the
    images are resampled with replacement for a 95% interval of PQ(model) - PQ(human)."""
    stems, per = [], []
    for stem, recs in sorted(records.items()):
        drawn = [r for r in recs if r]
        if len(drawn) < 2:
            continue
        human, model = Evaluation(), Evaluation()
        for i, pred in enumerate(drawn):
            for j, gt in enumerate(drawn):
                if j != i:
                    score_record(human, gt, pred)
                    score_record(model, gt, preds.get(stem, []))
        stems.append(stem)
        per.append((_counts(human), _counts(model)))
    if not per:
        return {"n_images": 0}
    h = np.array([p[0] for p in per])
    m = np.array([p[1] for p in per])
    out = {
        "n_images": len(stems),
        "human": _pq_sq_rq(h.sum(0)),
        "model": _pq_sq_rq(m.sum(0)),
    }
    if n_boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(stems), size=(n_boot, len(stems)))
        sh, sm = h[idx].sum(1), m[idx].sum(1)

        def pq(s):
            denom = s[:, 1] + 0.5 * s[:, 2] + 0.5 * s[:, 3]
            return np.where(denom > 0, s[:, 0] / np.maximum(denom, 1e-12), 0.0)

        d = pq(sm) - pq(sh)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["model_minus_human_pq_95"] = [round(float(lo), 4), round(float(hi), 4)]
        out["p_model_better"] = round(float((d > 0).mean()), 4)
        out["n_boot"], out["seed"] = n_boot, seed
    return out


def bootstrap(
    records: dict[str, list[list[dict]]],
    preds: dict[str, list[dict]],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Percentile 95% intervals of PQ, SQ and RQ, resampling images with replacement.

    Every record of a resampled image comes along with it, so the interval reflects which
    observations were held out, not which annotators drew them."""
    stems = sorted(records)
    per = np.zeros((len(stems), 4))  # sum of TP IoU, TP, FP, FN
    for i, stem in enumerate(stems):
        ev = evaluate({stem: records[stem]}, preds)
        per[i] = (sum(ev.tp_iou), ev.tp, ev.fp, ev.fn)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stems), size=(n_boot, len(stems)))
    s = per[idx].sum(axis=1)
    denom = s[:, 1] + 0.5 * s[:, 2] + 0.5 * s[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {
            "pq": np.where(denom > 0, s[:, 0] / denom, 0.0),
            "sq": np.where(s[:, 1] > 0, s[:, 0] / s[:, 1], 0.0),
            "rq": np.where(denom > 0, s[:, 1] / denom, 0.0),
        }
    out = {"n_boot": n_boot, "seed": seed, "n_images": len(stems)}
    for k, v in stats.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = [round(float(lo), 4), round(float(hi), 4)]
    return out
