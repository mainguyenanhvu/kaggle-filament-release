"""Synthetic checks of the training data path: the two deduplication rules, the split, the
augmentation, the seed-box jitter and the crop cache. Needs no competition data or weights.

  python -m unittest discover tests
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from filseg.data import (
    CropDataset,
    DetectorDataset,
    augment,
    build_crop_cache,
    collate,
    dedup_median_count,
    dedup_most_complete,
    instance_masks,
    jitter_bbox,
    load_coco,
    photometric,
    seed_box_canvas,
    split_stems,
)
from filseg.io import heldout_stems

SIZE = 256


def polygon(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def synthetic_coco(stems=("a", "b", "c"), records_per_stem=3):
    """One COCO dict: `records_per_stem` annotator records per frame, record k holding k+1
    rectangles, so the most-complete and the median-count rules pick different records."""
    images, annotations = [], []
    next_ann = 1
    for stem in stems:
        for k in range(records_per_stem):
            image_id = f"{stem}_r{k}"
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{stem}.jpeg",
                    "height": SIZE,
                    "width": SIZE,
                }
            )
            for j in range(k + 1):
                x0 = 20 + 40 * j
                annotations.append(
                    {
                        "id": next_ann,
                        "image_id": image_id,
                        "segmentation": polygon(x0, 30, x0 + 24, 90),
                        "area": 24 * 60,
                    }
                )
                next_ann += 1
    return {"images": images, "annotations": annotations}


def write_frames(directory: Path, stems, rng=np.random.default_rng(0)):
    for stem in stems:
        frame = rng.integers(40, 200, (SIZE, SIZE), dtype=np.uint8)
        frame[:10] = 0  # an off-disk strip, which the photometric jitter must not touch
        cv2.imwrite(str(directory / f"{stem}.jpeg"), frame)


class TestDedup(unittest.TestCase):
    def setUp(self):
        self.coco = synthetic_coco()

    def test_most_complete_takes_the_largest_record(self):
        chosen = dedup_most_complete(self.coco)
        self.assertEqual(chosen["a"], "a_r2")  # 3 instances beats 2 and 1

    def test_most_complete_breaks_ties_on_area(self):
        coco = {
            "images": [
                {"id": "x_r0", "file_name": "x.jpeg", "height": SIZE, "width": SIZE},
                {"id": "x_r1", "file_name": "x.jpeg", "height": SIZE, "width": SIZE},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": "x_r0",
                    "segmentation": polygon(0, 0, 10, 10),
                    "area": 100,
                },
                {
                    "id": 2,
                    "image_id": "x_r1",
                    "segmentation": polygon(0, 0, 20, 20),
                    "area": 400,
                },
            ],
        }
        self.assertEqual(dedup_most_complete(coco)["x"], "x_r1")

    def test_median_count_takes_the_middle_record(self):
        chosen = dedup_median_count(self.coco)
        self.assertEqual(chosen["a"], "a_r1")  # counts 1, 2, 3 -> median 2
        self.assertNotEqual(chosen["a"], dedup_most_complete(self.coco)["a"])

    def test_single_record_frame_is_kept_by_both_rules(self):
        coco = synthetic_coco(stems=("solo",), records_per_stem=1)
        self.assertEqual(dedup_most_complete(coco)["solo"], "solo_r0")
        self.assertEqual(dedup_median_count(coco)["solo"], "solo_r0")


class TestSplit(unittest.TestCase):
    def test_split_is_disjoint_and_matches_the_inference_split(self):
        coco = synthetic_coco(stems=tuple(f"s{i:03d}" for i in range(100)))
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "ann.json"
            ann.write_text(json.dumps(coco))
            train, val = split_stems(coco, n_val=10)
            self.assertEqual(len(val), 10)
            self.assertEqual(len(train), 90)
            self.assertFalse(set(train) & set(val))
            # the trainers hold out exactly the frames scripts/rubric_metrics.py --heldout scores
            self.assertEqual(set(val), heldout_stems(ann, n=10))
            self.assertEqual(load_coco(ann)["images"][0]["id"], coco["images"][0]["id"])

    def test_split_is_reproducible_for_a_seed(self):
        coco = synthetic_coco(stems=tuple(f"s{i:03d}" for i in range(50)))
        self.assertEqual(split_stems(coco, 5, 0), split_stems(coco, 5, 0))
        self.assertNotEqual(split_stems(coco, 5, 0)[1], split_stems(coco, 5, 1)[1])


class TestAugmentation(unittest.TestCase):
    def test_geometry_is_applied_to_image_and_masks_together(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 30:50] = 1
        marked = image.copy()
        marked[mask.astype(bool)] = 255  # a cue that must stay under the mask
        out_image, out_masks = augment(marked, [mask], random.Random(3))
        self.assertEqual(out_masks[0].sum(), mask.sum())
        covered = out_image[out_masks[0].astype(bool)]
        self.assertGreater((covered == covered.max()).mean(), 0.5)

    def test_photometric_leaves_the_off_disk_background_byte_identical(self):
        rng = np.random.default_rng(2)
        image = rng.integers(60, 200, (48, 48, 3), dtype=np.uint8)
        image[:12] = 0
        on_disk = image.max(axis=2) > 3
        for seed in range(8):
            out = photometric(image, on_disk, random.Random(seed))
            self.assertTrue((out[:12] == 0).all())
            self.assertEqual(out.dtype, np.uint8)

    def test_augmentation_is_reproducible_for_a_seed(self):
        image = np.random.default_rng(4).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 5:25] = 1
        first = augment(image, [mask], random.Random(7))
        second = augment(image, [mask], random.Random(7))
        self.assertTrue((first[0] == second[0]).all())
        self.assertTrue((first[1][0] == second[1][0]).all())


class TestSeedBox(unittest.TestCase):
    def test_jitter_stays_within_the_declared_bounds(self):
        rng = random.Random(0)
        for _ in range(200):
            x0, y0, x1, y1 = jitter_bbox((100.0, 200.0, 180.0, 260.0), rng)
            self.assertAlmostEqual((x1 - x0) / 80.0, 1.0, delta=0.16)
            self.assertAlmostEqual((y1 - y0) / 60.0, 1.0, delta=0.16)
            self.assertLessEqual(abs((x0 + x1) / 2 - 140.0), 0.10 * 80 + 0.08 * 80)

    def test_seed_canvas_is_clipped_to_the_frame(self):
        canvas = seed_box_canvas((-20.0, -10.0, 30.0, 40.0), 64, 64)
        self.assertEqual(canvas.shape, (64, 64))
        self.assertTrue((canvas[:40, :30] == 255).all())
        self.assertEqual(canvas[41:, 31:].sum(), 0)


class TestDatasets(unittest.TestCase):
    def test_detector_dataset_target_shapes(self):
        coco = synthetic_coco()
        with tempfile.TemporaryDirectory() as tmp:
            write_frames(Path(tmp), ["a", "b", "c"])
            train, _ = split_stems(coco, n_val=1)
            dataset = DetectorDataset(
                coco, train, dedup_most_complete(coco), tmp, train_aug=True
            )
            self.assertEqual(len(dataset), len(train))
            image, target = dataset[0]
            self.assertEqual(image.shape, (3, SIZE, SIZE))
            self.assertEqual(target["masks"].shape[0], target["boxes"].shape[0])
            self.assertEqual(target["boxes"].shape[1], 4)
            self.assertTrue((target["labels"] == 1).all())
            images, targets = collate([dataset[0], dataset[1]])
            self.assertEqual(len(images), 2)
            self.assertEqual(len(targets), 2)

    def test_crop_cache_covers_every_instance_and_feeds_three_channels(self):
        coco = synthetic_coco()
        chosen = dedup_median_count(coco)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_frames(tmp, ["a", "b", "c"])
            expected = sum(
                len(
                    instance_masks(
                        [a for a in coco["annotations"] if a["image_id"] == chosen[s]],
                        SIZE,
                        SIZE,
                    )
                )
                for s in ["a", "b"]
            )
            images, masks, n = build_crop_cache(
                coco,
                ["a", "b"],
                chosen,
                tmp,
                tmp / "cache",
                "train",
                random.Random(0),
                crop_size=64,
            )
            self.assertEqual(n, expected)
            cached = np.load(images)
            self.assertEqual(cached.shape, (n, 3, 64, 64))
            self.assertTrue(
                set(np.unique(cached[:, 2])) <= {0, 255}
            )  # seed box is binary
            self.assertGreater(cached[:, 2].max(), 0)
            self.assertEqual(np.load(masks).shape, (n, 64, 64))

            dataset = CropDataset(images, masks, augment=True)
            image, mask = dataset[0]
            self.assertEqual(image.shape, (3, 64, 64))
            self.assertEqual(mask.shape, (1, 64, 64))
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertEqual(sorted(torch.unique(mask).tolist()), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
