"""End-to-end inference: H-alpha image -> disjoint filament instance masks -> submission rows.

Per image:
  1. detector on the image and on a copy shifted by (16, 16) px (translate TTA); the shifted
     masks are shifted back and pooled with the unshifted ones;
  2. greedy mask NMS at IoU 0.1 over the pooled detections;
  3. detections scoring >= 0.80 are kept in score order, each taking only pixels no
     higher-scoring detection owns; masks under 250 px are dropped;
  4. each surviving mask is re-segmented by the seed-box crop U-Net (probability >= 0.40);
  5. refined masks take pixels in the same order, so the output instances never overlap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torchvision.transforms.functional import to_tensor


@dataclass(frozen=True)
class Config:
    score_thresh: float = 0.80  # detector score a detection needs to be kept
    refine_thresh: float = 0.40  # refiner probability a pixel needs to be filament
    min_area: int = 250  # pixels; smaller detections are dropped before refinement
    nms_iou: float = 0.10  # mask IoU at which a lower-scoring detection is a duplicate
    tta_shift: int = 16  # pixels; 0 disables the translate TTA
    crop_size: int = 256  # refiner input side
    crop_context: float = 1.8  # crop side = context x the longer bbox side
    crop_min_side: int = 96


CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    arr = np.array(img)
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    out[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = arr[
        max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return Image.fromarray(out)


def shift_mask_back(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Undo shift_image(dx, dy) on a mask predicted from the shifted image; pixels that came from
    the zero padding are dropped."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    out[max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)] = mask[
        max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)
    ]
    return out


@torch.no_grad()
def detect(
    detector, img: Image.Image, device, shift: int
) -> tuple[np.ndarray, np.ndarray]:
    """Scores and binary masks (mask probability > 0.5), pooled over the plain and shifted views."""
    out = detector([to_tensor(img).to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = (out["masks"][:, 0] > 0.5).cpu().numpy().astype(bool)
    if not shift:
        return scores, masks
    out_s = detector([to_tensor(shift_image(img, shift, shift)).to(device)])[0]
    scores_s = out_s["scores"].cpu().numpy()
    if not len(scores_s):
        return scores, masks
    masks_s = (out_s["masks"][:, 0] > 0.5).cpu().numpy().astype(bool)
    masks_s = np.stack([shift_mask_back(m, shift, shift) for m in masks_s])
    return np.concatenate([scores, scores_s]), np.concatenate([masks, masks_s])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def nms(scores: np.ndarray, masks: np.ndarray, iou_thresh: float):
    kept: list[int] = []
    for i in np.argsort(-scores):
        if all(mask_iou(masks[i], masks[j]) < iou_thresh for j in kept):
            kept.append(i)
    return scores[kept], masks[kept] if kept else masks[:0]


def assign_pixels(
    scores: np.ndarray, masks: np.ndarray, thresh: float
) -> list[np.ndarray]:
    """Keep detections scoring >= thresh (or the single best one if none does), in score order,
    each restricted to pixels not yet owned; empty remainders are dropped."""
    keep = [j for j in range(len(scores)) if scores[j] >= thresh]
    if not keep and len(scores):
        keep = [int(np.argmax(scores))]
    keep.sort(key=lambda j: scores[j], reverse=True)
    owned = np.zeros(masks.shape[1:], dtype=bool) if len(masks) else None
    out = []
    for j in keep:
        m = masks[j] & ~owned
        if m.any():
            owned |= m
            out.append(m)
    return out


def drop_small(masks: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    """Drop masks under min_area pixels, but keep the largest one if all are small."""
    kept = [m for m in masks if m.sum() >= min_area]
    if not kept and masks:
        kept = [max(masks, key=lambda m: m.sum())]
    return kept


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def square_crop(bbox, height: int, width: int, context: float, minimum: int):
    """A square window centred on bbox, context x its longer side (at least `minimum`), shifted
    to lie inside the image."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(np.ceil(max(x1 - x0, y1 - y0) * context))
    side = min(max(side, minimum), min(height, width))
    x0c, y0c = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1c, y1c = x0c + side, y0c + side
    if x0c < 0:
        x1c, x0c = x1c - x0c, 0
    if y0c < 0:
        y1c, y0c = y1c - y0c, 0
    if x1c > width:
        x0c, x1c = x0c - (x1c - width), width
    if y1c > height:
        y0c, y1c = y0c - (y1c - height), height
    return int(x0c), int(y0c), int(x1c), int(y1c)


@torch.no_grad()
def refine(
    refiner, gray: np.ndarray, clahe: np.ndarray, mask: np.ndarray, cfg: Config, device
) -> np.ndarray:
    """Re-segment one detection inside a square crop around it; falls back to the detector mask
    if the refiner returns nothing."""
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
    prob = refiner(x).sigmoid()[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = prob >= cfg.refine_thresh
    return out if out.any() else mask


@torch.no_grad()
def predict_image(
    path: Path, detector, refiner, cfg: Config, device
) -> list[np.ndarray]:
    """Disjoint boolean instance masks for one image, highest detector score first."""
    img = Image.open(path).convert("RGB")
    h, w = img.size[1], img.size[0]
    scores, masks = detect(detector, img, device, cfg.tta_shift)
    scores, masks = nms(scores, masks, cfg.nms_iou)
    masks = drop_small(assign_pixels(scores, masks, cfg.score_thresh), cfg.min_area)
    if not masks:
        return [], (h, w)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    clahe = CLAHE.apply(gray)
    owned = np.zeros((h, w), dtype=bool)
    out = []
    for m in masks:
        m = refine(refiner, gray, clahe, m, cfg, device) & ~owned
        if m.any():
            owned |= m
            out.append(m)
    return out, (h, w)


def encode(mask: np.ndarray) -> str:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def predict_all(paths, detector, refiner, cfg: Config, device, log_every: int = 25):
    """Yield (stem, masks, image shape, seconds) per image."""
    for i, path in enumerate(paths, start=1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        masks, shape = predict_image(path, detector, refiner, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i <= 3 or i % log_every == 0 or i == len(paths):
            print(
                f"[{i:03d}/{len(paths)}] {path.name}: {len(masks)} masks, {dt:.2f} s",
                flush=True,
            )
        yield path.stem, masks, shape, dt


def submission_rows(stem: str, masks: list[np.ndarray], shape: tuple[int, int]):
    """CSV rows `<stem>_<n>,<rle>`; an image with no detection gets one single-pixel mask so that
    every test image appears in the file."""
    if not masks:
        tiny = np.zeros(shape, dtype=bool)
        tiny[shape[0] // 2, shape[1] // 2] = True
        masks = [tiny]
    return [f"{stem}_{n},{encode(m)}" for n, m in enumerate(masks, start=1)]
