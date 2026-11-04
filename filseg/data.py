"""Training data: the annotation split, the two deduplication rules, the detector's augmentation
and the refiner's crop cache.

The organisers' COCO file holds one `images` entry per (annotator, image) record, so a physical
H-alpha frame appears once per annotator who traced it. Training uses one record per frame, and the
two networks pick it by different rules, which is deliberate and is kept here as it was measured:

* the detector takes the **most complete** record (most instances, ties by largest total mask area),
  on the argument that a larger tracing of this data is usually a superset of a smaller one;
* the refiner takes the **median-count** record, the earlier rule, because its crops are per
  instance and the extra instances of a maximal tracing are the faintest and least agreed ones.

Evaluation never deduplicates: `filseg.metrics` scores every record (see `scripts/rubric_metrics.py`).
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import to_tensor

from filseg.infer import CLAHE, bbox_from_mask, square_crop

# --------------------------------------------------------------------------- annotations


def load_coco(ann_path: str | Path) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def _records_by_stem(coco: dict) -> dict[str, list]:
    by_stem: dict[str, list] = {}
    for im in coco["images"]:
        by_stem.setdefault(Path(im["file_name"]).stem, []).append(im["id"])
    return by_stem


def _instance_counts(coco: dict) -> tuple[dict, dict]:
    counts: dict = {}
    areas: dict = {}
    for ann in coco["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
        areas[ann["image_id"]] = areas.get(ann["image_id"], 0.0) + float(
            ann.get("area", 0.0)
        )
    return counts, areas


def dedup_most_complete(coco: dict) -> dict[str, str]:
    """Image stem -> the record with the most instances (ties: largest total mask area). The
    detector's rule."""
    counts, areas = _instance_counts(coco)
    chosen = {}
    for stem, ids in _records_by_stem(coco).items():
        chosen[stem] = (
            ids[0]
            if len(ids) == 1
            else max(ids, key=lambda i: (counts.get(i, 0), areas.get(i, 0.0)))
        )
    return chosen


def dedup_median_count(coco: dict) -> dict[str, str]:
    """Image stem -> the record whose instance count is closest to the stem's median count (ties:
    the smaller count, then the earlier record). The refiner's rule."""
    counts, _ = _instance_counts(coco)
    chosen = {}
    for stem, ids in _records_by_stem(coco).items():
        if len(ids) == 1:
            chosen[stem] = ids[0]
            continue
        median = statistics.median([counts.get(i, 0) for i in ids])
        chosen[stem] = min(ids, key=lambda i: abs(counts.get(i, 0) - median))
    return chosen


def split_stems(
    coco: dict, n_val: int = 60, seed: int = 0
) -> tuple[list[str], list[str]]:
    """(train, validation) image stems. The shuffle is `random.seed(seed); random.shuffle(...)` over
    the sorted stems, as in development, so the 60 held-out frames are the same set that
    `filseg.io.heldout_stems` returns and that every number in the report is measured on."""
    stems = sorted({Path(im["file_name"]).stem for im in coco["images"]})
    random.seed(seed)
    random.shuffle(stems)
    return stems[n_val:], stems[:n_val]


def instance_masks(anns: list[dict], height: int, width: int) -> list[np.ndarray]:
    """One {0,1} uint8 mask per annotation; a multi-polygon annotation is merged into one mask."""
    return [
        mask_util.decode(
            mask_util.merge(mask_util.frPyObjects(a["segmentation"], height, width))
        )
        for a in anns
    ]


def group_annotations(coco: dict) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        by_id.setdefault(ann["image_id"], []).append(ann)
    return by_id


# --------------------------------------------------------------------------- detector


ON_DISK_LEVEL = 3  # off-disk background is ~pure black; above this is solar disk


def photometric(img: np.ndarray, on_disk: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gamma, gain/bias, blur and noise, applied only inside `on_disk`. The off-disk background
    stays byte-identical: brightening it would teach the detector a cue the test frames lack."""
    out = img.astype(np.float32)
    if rng.random() < 0.70:
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, rng.uniform(0.80, 1.25))
    if rng.random() < 0.70:
        out = out * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
    out = np.clip(out, 0, 255)
    if rng.random() < 0.15:
        out = cv2.GaussianBlur(out, (5, 5), sigmaX=rng.uniform(0.5, 1.5))
    if rng.random() < 0.25:
        out = out + np.random.normal(0, rng.uniform(3, 12), size=out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.where(on_disk[..., None], out, img)


def augment(
    img: np.ndarray, masks: list[np.ndarray], rng: random.Random | None = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random 90-degree rotation, horizontal flip and photometric jitter. Geometry is applied to the
    image and every mask together; `img` is HWC uint8 and each mask is HW uint8 {0,1}."""
    rng = rng or random
    on_disk = img.max(axis=2) > ON_DISK_LEVEL

    k = rng.choice([0, 1, 2, 3])
    if k:
        img = np.rot90(img, k, axes=(0, 1)).copy()
        on_disk = np.rot90(on_disk, k, axes=(0, 1)).copy()
        masks = [np.rot90(m, k, axes=(0, 1)).copy() for m in masks]
    if rng.random() < 0.5:
        img = np.fliplr(img).copy()
        on_disk = np.fliplr(on_disk).copy()
        masks = [np.fliplr(m).copy() for m in masks]
    return photometric(img, on_disk, rng), masks


class DetectorDataset(Dataset):
    """(image tensor, Mask R-CNN target) per training frame: the deduplicated record's instances as
    masks plus their tight boxes. Instances whose mask is empty after augmentation are dropped."""

    def __init__(
        self,
        coco: dict,
        stems: list[str],
        chosen_id: dict[str, str],
        image_dir: str | Path,
        train_aug: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.train_aug = train_aug
        meta_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_id = group_annotations(coco)
        self.items = []
        for stem in stems:
            meta = meta_by_id[chosen_id[stem]]
            anns = anns_by_id.get(chosen_id[stem], [])
            if anns:
                self.items.append((stem, meta, anns))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self._load(index, *self.items[index])

    def _load(self, index: int, stem: str, meta: dict, anns: list[dict]):
        img = np.array(Image.open(self.image_dir / meta["file_name"]).convert("RGB"))
        masks = instance_masks(anns, meta["height"], meta["width"])
        if self.train_aug:
            img, masks = augment(img, masks)

        boxes, kept = [], []
        for m in masks:
            ys, xs = m.nonzero()
            if len(xs) == 0:
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            kept.append(m)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(kept), dtype=torch.uint8),
            "image_id": index,
            "stem": stem,
        }
        return to_tensor(img), target


def annotator_of(image_id: str) -> str:
    """The annotator part of a record id `<annotator>-<stem>`."""
    return image_id.split("-")[0]


def annotator_index(coco: dict) -> dict[str, int]:
    """Annotator id -> row of the FiLM embeddings, over every record in the annotation file."""
    return {a: i for i, a in enumerate(sorted({annotator_of(im["id"]) for im in coco["images"]}))}


class AllRecordsDataset(DetectorDataset):
    """One item per (training frame, non-empty annotator record); the target carries `annotator`.
    The annotator-conditioned detector trains on this, so each record is self-consistent given its
    id instead of one frame carrying contradictory labels."""

    def __init__(self, coco: dict, stems: list[str], image_dir: str | Path, train_aug: bool = False):
        self.image_dir = Path(image_dir)
        self.train_aug = train_aug
        anns_by_id = group_annotations(coco)
        index = annotator_index(coco)
        keep = set(stems)
        self.items, self.annotators = [], []
        for meta in sorted(coco["images"], key=lambda r: r["id"]):
            stem = Path(meta["file_name"]).stem
            anns = anns_by_id.get(meta["id"], [])
            if stem in keep and anns:
                self.items.append((stem, meta, anns))
                self.annotators.append(index[annotator_of(meta["id"])])

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target["annotator"] = self.annotators[index]
        return image, target


class RandomRecordDataset(DetectorDataset):
    """One item per training frame; each access draws one of its non-empty records uniformly, so the
    target over the epochs is the expected annotator. Gradient steps per epoch match the one-record
    recipe."""

    def __init__(self, coco: dict, stems: list[str], image_dir: str | Path, train_aug: bool = False):
        self.image_dir = Path(image_dir)
        self.train_aug = train_aug
        anns_by_id = group_annotations(coco)
        records: dict[str, list] = {}
        for meta in coco["images"]:
            anns = anns_by_id.get(meta["id"], [])
            if anns:
                records.setdefault(Path(meta["file_name"]).stem, []).append((meta, anns))
        self.records = [(s, records[s]) for s in stems if s in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        stem, recs = self.records[index]
        return self._load(index, stem, *random.choice(recs))


class SameAnnotatorBatches(Sampler):
    """Batches of one annotator each (FiLM reads one id per forward pass), reshuffled per epoch."""

    def __init__(self, annotators: list[int], batch_size: int, seed: int = 0):
        self.annotators, self.batch_size, self.seed, self.epoch = annotators, batch_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups: dict[int, list[int]] = {}
        for i, a in enumerate(self.annotators):
            groups.setdefault(a, []).append(i)
        batches = []
        for members in groups.values():
            rng.shuffle(members)
            batches += [members[k : k + self.batch_size] for k in range(0, len(members), self.batch_size)]
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- refiner


SCALE_JITTER = (
    0.15  # the detector's boxes are not tight, so the refiner trains on jittered boxes
)
TRANSLATE_JITTER = 0.10


def jitter_bbox(bbox, rng: random.Random):
    """Scale each side by 1 +- 15% and translate the centre by +- 10% of the box size, simulating a
    detector box. The refiner never sees a ground-truth box at inference, so it never trains on one.
    The four draws are taken in the order scale-x, scale-y, shift-x, shift-y, so a given seed
    reproduces the crop cache the submitted refiner was trained on."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    w2 = w * (1 + rng.uniform(-SCALE_JITTER, SCALE_JITTER))
    h2 = h * (1 + rng.uniform(-SCALE_JITTER, SCALE_JITTER))
    cx = (x0 + x1) / 2 + rng.uniform(-TRANSLATE_JITTER, TRANSLATE_JITTER) * w
    cy = (y0 + y1) / 2 + rng.uniform(-TRANSLATE_JITTER, TRANSLATE_JITTER) * h
    return cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2


def seed_box_canvas(bbox, height: int, width: int) -> np.ndarray:
    """The seed box drawn as a filled rectangle at full image resolution (the refiner's 3rd channel).
    `filseg.infer.refine` builds the same canvas from the detector's mask bbox."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = max(int(round(bbox[0])), 0), max(int(round(bbox[1])), 0)
    x1, y1 = min(int(round(bbox[2])), width), min(int(round(bbox[3])), height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = 255
    return canvas


def build_crop_cache(
    coco: dict,
    stems: list[str],
    chosen_id: dict[str, str],
    image_dir: str | Path,
    out_dir: str | Path,
    prefix: str,
    rng: random.Random,
    crop_size: int = 256,
    context: float = 1.8,
    min_side: int = 96,
) -> tuple[Path, Path, int]:
    """Write `<prefix>_crop_images3ch.npy` (N,3,S,S uint8) and `<prefix>_crop_masks.npy` (N,S,S) for
    every instance of every chosen record: channels [grey, CLAHE(grey), seed box], exactly the three
    `filseg.infer.refine` feeds at inference. Returns the two paths and N."""
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_id = group_annotations(coco)

    rows = [
        (stem, meta_by_id[chosen_id[stem]], anns_by_id.get(chosen_id[stem], []))
        for stem in stems
        if anns_by_id.get(chosen_id[stem])
    ]
    total = sum(len(anns) for _, _, anns in rows)
    image_path = out_dir / f"{prefix}_crop_images3ch.npy"
    mask_path = out_dir / f"{prefix}_crop_masks.npy"
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(total, 3, crop_size, crop_size)
    )
    masks = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=(total, crop_size, crop_size)
    )

    size = (crop_size, crop_size)
    position = 0
    for stem, meta, anns in rows:
        grey = cv2.imread(str(image_dir / meta["file_name"]), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise FileNotFoundError(image_dir / meta["file_name"])
        h, w = meta["height"], meta["width"]
        clahe = CLAHE.apply(grey)
        for m in instance_masks(anns, h, w):
            if m.sum() == 0:
                continue
            seed = jitter_bbox(bbox_from_mask(m), rng)
            x0, y0, x1, y1 = square_crop(seed, h, w, context, min_side)
            canvas = seed_box_canvas(seed, h, w)
            images[position, 0] = cv2.resize(
                grey[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
            )
            images[position, 1] = cv2.resize(
                clahe[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
            )
            images[position, 2] = cv2.resize(
                canvas[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST
            )
            masks[position] = cv2.resize(
                m[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST
            )
            position += 1
        print(f"cached {position}/{total} crops ({stem})", flush=True)

    images.flush()
    masks.flush()
    if position != total:
        # every annotation of a chosen record must have produced one crop
        raise AssertionError(f"{prefix}: expected {total} crops, wrote {position}")
    return image_path, mask_path, position


class CropDataset(Dataset):
    """A cached crop and its mask, as float tensors in [0, 1]. With `augment=True`: random 90-degree
    rotation, horizontal flip, and gamma/gain on the two image channels but never on the seed box."""

    def __init__(self, image_path, mask_path, augment: bool = False):
        self.images = np.load(image_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).float() / 255
        mask = (
            torch.from_numpy(np.asarray(self.masks[index]).copy()).float().unsqueeze(0)
        )
        if self.augment:
            k = random.randrange(4)
            image, mask = torch.rot90(image, k, (1, 2)), torch.rot90(mask, k, (1, 2))
            if random.random() < 0.5:
                image, mask = torch.flip(image, (2,)), torch.flip(mask, (2,))
            if random.random() < 0.5:
                image[:2] = image[:2] ** random.uniform(0.85, 1.20)
            if random.random() < 0.5:
                image[:2] = torch.clamp(image[:2] * random.uniform(0.90, 1.10), 0, 1)
        return image, mask
