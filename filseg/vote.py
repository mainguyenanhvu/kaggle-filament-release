"""Class-balanced object consensus over an existing instance list.

Link each base instance to the highest-IoU mask from each voter (IoU > 0.4).
The base has one vote; linked voters in the same class share one vote in total.
Keep pixels with weighted support >= 0.5, in greedy disjoint ownership order.
Keep the base shape when it or the proposed consensus has fewer than 250 pixels.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

MA = 250
LINK = 0.4
VOTE = 0.5


def iou_mat(a: list[dict], b: list[dict]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mask_util.iou(a, b, [0] * len(b))).reshape(len(a), len(b))


def cb_vote(
    base_rles: list[dict],
    voters: dict[str, list[dict]],
    classes: dict[str, str] | None = None,
    link: float = LINK,
    vote: float = VOTE,
) -> list[dict]:
    """Reshape base objects without adding or dropping instances.

    Class weights count only voters linked to the current object. Unspecified
    voters each have their own class; the base always counts as a separate vote.
    With no voters, return the base list unchanged.
    """
    if not base_rles:
        return []
    if not voters:
        return list(base_rles)
    classes = classes or {}
    h, w = base_rles[0]["size"]
    owned = np.zeros((h, w), bool)
    out = []
    for r in base_rles:
        m = mask_util.decode(r).astype(bool) & ~owned
        me = mask_util.encode(np.asarray(m, np.uint8, order="F"))
        if m.sum() < MA:
            owned |= m
            out.append(me)
            continue
        linked = {}
        for name, rles in voters.items():
            if not rles:
                continue
            iou = iou_mat([me], rles)[0]
            j = int(np.argmax(iou))
            if iou[j] > link:
                linked[name] = rles[j]
        if not linked:
            owned |= m
            out.append(me)
            continue
        labels = {
            name: ("class", classes[name]) if name in classes else ("voter", name)
            for name in linked
        }
        counts = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        acc, tot = m.astype(np.float32), 1.0
        for name, r2 in linked.items():
            weight = 1.0 / counts[labels[name]]
            acc += mask_util.decode(r2).astype(np.float32) * weight
            tot += weight
        nm = (acc / tot >= vote - 1e-9) & ~owned
        if nm.sum() < MA:
            owned |= m
            out.append(me)
            continue
        owned |= nm
        out.append(mask_util.encode(np.asarray(nm, np.uint8, order="F")))
    return out
