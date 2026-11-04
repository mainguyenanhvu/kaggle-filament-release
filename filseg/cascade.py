"""Cached candidates of a detector and the masks a vote member reads from them.

The record vote's cascade member is not a finished submission: it is the detector's candidate list
(every detection above a 0.30 floor, owned in score order, each with the refiner's 256 x 256
probability map and its crop bounds), turned into masks at score >= 0.80, refiner probability
>= 0.40 and area >= 250 px, which are then one more voter of `filseg.vote.cb_vote`.

  candidates(...)  -> {stem: (height, width, [(score, prob256, (x0, y0, x1, y1)), ...])}
  member_masks(...) -> list of COCO run-length masks, disjoint, in score order
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from .infer import (
    CLAHE,
    Config,
    bbox_from_mask,
    detect,
    drop_small,
    nms,
    square_crop,
)

SCORE_FLOOR = 0.3  # candidates below this detector score are never cached


@torch.no_grad()
def refiner_prob(refiner, gray, clahe, mask, cfg: Config, device):
    """The refiner's probability map (crop_size x crop_size) and square crop bounds of one mask."""
    h, w = mask.shape
    tight = bbox_from_mask(mask)
    x0, y0, x1, y1 = square_crop(tight, h, w, cfg.crop_context, cfg.crop_min_side)
    seed = np.zeros((h, w), dtype=np.uint8)
    sx0, sy0 = max(int(round(tight[0])), 0), max(int(round(tight[1])), 0)
    sx1, sy1 = min(int(round(tight[2])), w), min(int(round(tight[3])), h)
    if sx1 > sx0 and sy1 > sy0:
        seed[sy0:sy1, sx0:sx1] = 255
    size = (cfg.crop_size, cfg.crop_size)
    channels = [
        cv2.resize(gray[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(clahe[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(seed[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST),
    ]
    x = torch.from_numpy(np.stack(channels)).float().div(255).unsqueeze(0).to(device)
    return refiner(x).sigmoid()[0, 0].cpu().numpy(), (x0, y0, x1, y1)


@torch.no_grad()
def image_candidates(path: Path, detector, refiner, cfg: Config, device):
    img = Image.open(path).convert("RGB")
    h, w = img.size[1], img.size[0]
    scores, masks = detect(detector, img, device, cfg.tta_shift)
    scores, masks = nms(scores, masks, cfg.nms_iou)
    keep = [j for j in range(len(scores)) if scores[j] >= SCORE_FLOOR]
    if not keep and len(scores):
        keep = [int(np.argmax(scores))]
    keep.sort(key=lambda j: scores[j], reverse=True)
    owned = np.zeros((h, w), dtype=bool)
    owned_masks = []
    for j in keep:
        m = masks[j] & ~owned
        if m.sum() == 0:
            continue
        owned |= m
        owned_masks.append((float(scores[j]), m))
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    clahe = CLAHE.apply(gray)
    return h, w, [(sc, *refiner_prob(refiner, gray, clahe, m, cfg, device)) for sc, m in owned_masks]


def candidates(paths, detector, refiner, cfg: Config, device) -> dict:
    return {
        p.stem: image_candidates(p, detector, refiner, cfg, device) for p in paths
    }


def refine_threshold(prob, bounds, h, w, threshold):
    x0, y0, x1, y1 = bounds
    crop = cv2.resize(prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = crop >= threshold
    return out


def member_masks(entry, cfg: Config = Config()) -> list[dict]:
    """RLE masks of one image's cached candidates at cfg.score_thresh / refine_thresh / min_area."""
    h, w, cached = entry
    admitted = [c for c in cached if c[0] >= cfg.score_thresh]
    if not admitted and cached:
        admitted = [max(cached, key=lambda c: c[0])]
    owned = np.zeros((h, w), dtype=bool)
    raw = []
    for _, prob, bounds in admitted:
        m = refine_threshold(prob, bounds, h, w, cfg.refine_thresh) & ~owned
        if m.sum():
            owned |= m
            raw.append(m)
    raw = drop_small(raw, cfg.min_area)
    return [mask_util.encode(np.asarray(m, dtype=np.uint8, order="F")) for m in raw]
