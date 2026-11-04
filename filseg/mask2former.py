"""The Mask2Former voter: a box-free query segmenter (Cheng et al., CVPR 2022, arXiv:2112.01527).

Every instance is a query whose mask is a dot product with stride-4 pixel features over the whole
frame, so a long curved filament is never cut by a box. Inference reads the whole 2048 x 2048 frame;
training (`scripts/train_mask2former.py`) reads 1024 x 1024 full-resolution crops.

  load_mask2former(...) builds the network from a config file, downloading nothing.
  query_masks(...)      -> [(score, RLE), ...] best first, score = class probability x mean mask probability
  member_masks(...)     -> disjoint RLEs at score >= 0.90, first wins, area >= 250 px; a frame where
                           nothing passes keeps its best query
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_util
from torchvision.transforms.functional import to_tensor

PRETRAINED = "facebook/mask2former-swin-small-coco-instance"
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
SCORE_FLOOR = 0.05  # queries below this are never kept
SCORE_THRESH = 0.90  # chosen on 40 training-pool frames, never on the held-out split
MIN_AREA = 250


def normalise(image: torch.Tensor) -> torch.Tensor:
    return (image - MEAN.to(image.device)) / STD.to(image.device)


def build_pretrained():
    """The COCO-pretrained network with one class, for training (downloads the weights)."""
    from transformers import Mask2FormerForUniversalSegmentation

    return Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED, num_labels=1, ignore_mismatched_sizes=True
    )


def load_mask2former(weights: str | Path, config: str | Path, device):
    """Our trained network from its checkpoint and a saved config, with no download."""
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_json_file(str(config)))
    model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
    return model.to(device).eval()


@torch.no_grad()
def query_masks(model, path: str | Path, device, score_floor: float = SCORE_FLOOR):
    image = to_tensor(Image.open(path).convert("RGB"))
    height, width = image.shape[1:]
    pixels = normalise(image)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(pixel_values=pixels)
    class_prob = out.class_queries_logits[0].float().softmax(-1)[:, 0]
    logits = out.masks_queries_logits[0].float()
    probs = F.interpolate(
        logits[None], size=(height, width), mode="bilinear", align_corners=False
    )[0].sigmoid()
    found = []
    for q in range(probs.shape[0]):
        if class_prob[q] < score_floor:
            continue
        binary = probs[q] > 0.5
        if int(binary.sum()) < MIN_AREA:
            continue
        score = float(class_prob[q]) * float(probs[q][binary].mean())
        if score >= score_floor:
            rle = mask_util.encode(np.asarray(binary.cpu().numpy(), dtype=np.uint8, order="F"))
            found.append((score, rle))
    return sorted(found, key=lambda item: -item[0])


def member_masks(queries, threshold: float = SCORE_THRESH) -> list[dict]:
    kept = [(s, r) for s, r in queries if s >= threshold] or queries[:1]
    owned = None
    out = []
    for _, rle in kept:
        mask = mask_util.decode(rle).astype(bool)
        owned = np.zeros_like(mask) if owned is None else owned
        mask &= ~owned
        if mask.sum() >= MIN_AREA:
            owned |= mask
            out.append(mask_util.encode(np.asarray(mask, dtype=np.uint8, order="F")))
    return out
