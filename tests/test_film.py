"""The annotator-conditioned detector and its training sets, on synthetic data (no download)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from filseg.data import (
    AllRecordsDataset,
    RandomRecordDataset,
    SameAnnotatorBatches,
    annotator_index,
)
from filseg.model import AnnotatorFiLM, build_detector, load_detector, set_annotator


def coco_fixture(directory: Path) -> dict:
    """Two frames: one drawn by annotators 010101 and 020202, one drawn only by 020202. The
    empty record of a third annotator must never reach a training set."""
    polygon = [[10.0, 10.0, 50.0, 10.0, 50.0, 30.0, 10.0, 30.0]]
    images, annotations = [], []
    for stem, annotators in (("a", ("010101", "020202", "030303")), ("b", ("020202",))):
        Image.fromarray(np.full((64, 64, 3), 120, np.uint8)).save(directory / f"{stem}.jpeg")
        for annotator in annotators:
            record = f"{annotator}-{stem}"
            images.append({"id": record, "file_name": f"{stem}.jpeg", "height": 64, "width": 64})
            if annotator != "030303":
                annotations.append({"image_id": record, "segmentation": polygon, "area": 800})
    return {"images": images, "annotations": annotations}


class TestFiLM(unittest.TestCase):
    def test_zero_init_is_identity_and_rows_differ_once_set(self):
        film = AnnotatorFiLM(torch.nn.Identity(), 3, 4)
        x = torch.ones(2, 4)
        self.assertTrue(torch.equal(film(x), x))
        with torch.no_grad():
            film.gamma.weight[1] = 1.0
            film.beta.weight[3] = 5.0
        film.annotator = 1
        self.assertTrue(torch.equal(film(x), 2 * x))
        film.annotator = -1  # the null row is the last one
        self.assertTrue(torch.equal(film(x), x + 5))

    def test_spatial_output_broadcasts(self):
        film = AnnotatorFiLM(torch.nn.Identity(), 2, 8)
        self.assertEqual(film(torch.ones(3, 8, 7, 7)).shape, (3, 8, 7, 7))

    def test_cascade_refuses_film(self):
        with self.assertRaises(ValueError):
            build_detector(128, pretrained=False, cascade=True, film_ids=2)

    def test_checkpoint_roundtrip_sets_the_annotator(self):
        model = build_detector(128, pretrained=False, film_ids=5)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            torch.save({"model": model.state_dict()}, path)
            loaded = load_detector(path, "cpu", img_size=128, film_ids=5, annotator=3)
        films = [m for m in loaded.modules() if isinstance(m, AnnotatorFiLM)]
        self.assertEqual(len(films), 2)
        self.assertEqual({m.annotator for m in films}, {3})
        set_annotator(loaded, -1)
        self.assertEqual({m.annotator for m in films}, {-1})

    def test_film_changes_detector_output(self):
        model = build_detector(128, pretrained=False, film_ids=2).eval()
        with torch.no_grad():
            model.roi_heads.box_head.beta.weight[1] = 3.0
        x = torch.rand(2, 256, 7, 7)
        set_annotator(model, 0)
        a = model.roi_heads.box_head(x)
        set_annotator(model, 1)
        b = model.roi_heads.box_head(x)
        self.assertFalse(torch.equal(a, b))


class TestRecordDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.coco = coco_fixture(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_annotator_index_is_sorted_over_every_record(self):
        self.assertEqual(annotator_index(self.coco), {"010101": 0, "020202": 1, "030303": 2})

    def test_all_records_keeps_non_empty_records_with_their_annotator(self):
        ds = AllRecordsDataset(self.coco, ["a", "b"], self.dir)
        self.assertEqual(len(ds), 3)
        self.assertEqual(sorted(ds.annotators), [0, 1, 1])
        for i in range(len(ds)):
            _, target = ds[i]
            self.assertEqual(target["annotator"], ds.annotators[i])
            self.assertEqual(target["boxes"].shape, (1, 4))

    def test_all_records_excludes_other_stems(self):
        ds = AllRecordsDataset(self.coco, ["b"], self.dir)
        self.assertEqual([item[0] for item in ds.items], ["b"])

    def test_random_record_is_one_item_per_frame_and_reads_both_records(self):
        ds = RandomRecordDataset(self.coco, ["a", "b"], self.dir)
        self.assertEqual(len(ds), 2)
        drawn = set()
        for _ in range(40):
            image, target = ds[0]
            self.assertEqual(target["stem"], "a")
            drawn.add(json.dumps(target["boxes"].tolist()))
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(len(drawn), 1)  # both records hold the same polygon in this fixture

    def test_same_annotator_batches_cover_every_item_once(self):
        annotators = [0, 0, 0, 1, 1, 2, 0, 1]
        sampler = SameAnnotatorBatches(annotators, batch_size=2, seed=3)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(annotators))))
        for batch in batches:
            self.assertEqual(len({annotators[i] for i in batch}), 1)
            self.assertLessEqual(len(batch), 2)

    def test_same_annotator_batches_reshuffle_per_epoch(self):
        sampler = SameAnnotatorBatches(list(range(5)) * 8, batch_size=2, seed=0)
        first = list(sampler)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))


if __name__ == "__main__":
    unittest.main()
