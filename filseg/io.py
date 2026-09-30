"""Readers for the challenge annotations and for predictions.

Annotations are the organisers' COCO file: one `images` entry per (annotator, image) record,
keyed to the H-alpha frame by the stem of `file_name`, with polygon `segmentation`s. Predictions
are either the Kaggle submission CSV (`filament_id,segmentation_rle`, id `<stem>_<n>`) or a JSON
lines file with one `{"stem": ..., "masks": [<rle counts string>, ...]}` object per image.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from pycocotools import mask as mask_util

IMAGE_SIZE = 2048


def load_records(
    ann_path: str | Path, stems: set[str] | None = None
) -> dict[str, list[list[dict]]]:
    """Image stem -> list of annotator records, each a list of ground-truth RLEs."""
    coco = json.load(open(ann_path))
    by_image: dict = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    records: dict[str, list[list[dict]]] = {}
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        if stems is not None and stem not in stems:
            continue
        h, w = im["height"], im["width"]
        rles = [
            mask_util.merge(mask_util.frPyObjects(a["segmentation"], h, w))
            for a in by_image.get(im["id"], [])
        ]
        records.setdefault(stem, []).append(rles)
    return records


def all_stems(ann_path: str | Path) -> list[str]:
    coco = json.load(open(ann_path))
    return sorted({Path(im["file_name"]).stem for im in coco["images"]})


def heldout_stems(ann_path: str | Path, n: int = 60, seed: int = 0) -> set[str]:
    """The validation split used throughout development: the first n of the sorted training
    stems after a seeded shuffle. Every image stem (all its annotator records) lands on one side."""
    stems = all_stems(ann_path)
    random.seed(seed)
    random.shuffle(stems)
    return set(stems[:n])


def _rle(counts: str, size: int = IMAGE_SIZE) -> dict:
    return {"size": [size, size], "counts": counts.encode("ascii")}


def load_predictions(path: str | Path) -> dict[str, list[dict]]:
    """Image stem -> list of predicted RLEs, from a submission CSV or a JSON lines file."""
    path = Path(path)
    out: dict[str, list[dict]] = {}
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                stem = row["filament_id"].rsplit("_", 1)[0]
                out.setdefault(stem, []).append(_rle(row["segmentation_rle"]))
        return out
    for line in open(path):
        r = json.loads(line)
        out[r["stem"]] = [_rle(m) for m in r["masks"]]
    return out
