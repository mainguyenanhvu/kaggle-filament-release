"""The voted pipeline's plumbing on synthetic data (no weights, no download)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from filseg import ensemble, mask2former
from filseg.ensemble import Member, read_manifest, vote_image
from filseg.infer import Config


def rle(y0, y1, x0, x1, size=64):
    m = np.zeros((size, size), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asarray(m, order="F"))


MANIFEST = {
    "refiner": "refiner.pt",
    "base": {"name": "acond", "file": "acond.pt", "film_ids": 37, "annotator": 28},
    "voters": [
        {"name": "allrec", "file": "allrec.pt", "class": "r50"},
        {"name": "m2f", "file": "m2f.pt", "kind": "mask2former", "config": "c.json"},
        {"name": "casc", "file": "c.pt", "backbone": "convnext_tiny.dinov3_lvd1689m", "cascade": True},
    ],
}


class TestManifest(unittest.TestCase):
    def test_fields_reach_the_members(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vote.json"
            path.write_text(json.dumps(MANIFEST))
            base, voters, refiner = read_manifest(path)
        self.assertEqual(refiner, "refiner.pt")
        self.assertEqual((base.film_ids, base.annotator), (37, 28))
        self.assertEqual(voters[0].voter_class, "r50")
        self.assertEqual((voters[1].kind, voters[1].config), ("mask2former", "c.json"))
        self.assertTrue(voters[2].cascade)

    def test_unknown_field_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vote.json"
            bad = dict(MANIFEST, base={"name": "a", "file": "a.pt", "typo": 1})
            path.write_text(json.dumps(bad))
            with self.assertRaises(TypeError):
                read_manifest(path)


class TestVoteImage(unittest.TestCase):
    def test_voters_reshape_the_base_and_keep_its_instance_count(self):
        base_mask = rle(10, 40, 10, 50)
        grown = rle(10, 40, 10, 60)  # both voters extend the object by the same 10 columns
        entries = {"base": [base_mask], "a": [grown], "b": [grown]}

        def candidates(path, network, refiner, cfg, device):
            return network

        with (
            mock.patch.object(ensemble, "image_candidates", candidates),
            mock.patch.object(ensemble, "member_masks", lambda entry, cfg: entries[entry]),
        ):
            out = vote_image(
                Path("x.jpeg"),
                (Member("base", "b.pt"), "base"),
                [(Member("a", "a.pt"), "a"), (Member("b", "b.pt"), "b")],
                None,
                Config(),
                "cpu",
            )
        self.assertEqual(len(out), 1)
        # two of three votes clear the 0.5 support, so the extension survives
        self.assertEqual(int(mask_util.area(out[0])), int(mask_util.area(grown)))

    def test_mask2former_voter_reads_its_own_path(self):
        with (
            mock.patch.object(ensemble, "image_candidates", lambda *a, **k: "base"),
            mock.patch.object(ensemble, "member_masks", lambda entry, cfg: [rle(10, 40, 10, 50)]),
            mock.patch.object(mask2former, "query_masks", return_value=[(0.95, rle(10, 40, 10, 50))]) as q,
        ):
            vote_image(
                Path("x.jpeg"),
                (Member("base", "b.pt"), "net"),
                [(Member("m", "m.pt", kind="mask2former"), "m2f-net")],
                None,
                Config(),
                "cpu",
            )
        self.assertEqual(q.call_args.args[0], "m2f-net")


class FakeMask2Former:
    """Three queries on an 8 x 8 logit grid: a good one, one with a low class score, one too small."""

    def __call__(self, pixel_values):
        h = pixel_values.shape[-1] // 4
        masks = torch.full((1, 3, h, h), -10.0)
        masks[0, 0, 2:6, 2:6] = 10.0
        masks[0, 1, 2:6, 2:6] = 10.0
        masks[0, 2, 0, 0] = 10.0
        classes = torch.tensor([[[6.0, 0.0], [-6.0, 0.0], [6.0, 0.0]]])
        return SimpleNamespace(class_queries_logits=classes, masks_queries_logits=masks)


class TestQueryMasks(unittest.TestCase):
    def test_floor_area_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.png"
            Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(path)
            found = mask2former.query_masks(FakeMask2Former(), path, torch.device("cpu"))
        self.assertEqual(len(found), 1)  # the low class score and the small mask are gone
        score, rle_mask = found[0]
        self.assertGreater(score, 0.9)
        self.assertGreater(int(mask_util.area(rle_mask)), 250)


class TestVoteAll(unittest.TestCase):
    def test_loads_each_network_once_and_yields_per_image(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "vote.json"
            manifest.write_text(json.dumps(MANIFEST))
            paths = [Path(d) / "a.jpeg", Path(d) / "b.jpeg"]
            with (
                mock.patch.object(ensemble, "load_refiner", return_value="refiner"),
                mock.patch.object(ensemble, "load_member", return_value="net") as load,
                mock.patch.object(ensemble, "vote_image", return_value=[rle(10, 40, 10, 50)]) as vote,
            ):
                out = list(ensemble.vote_all(paths, manifest, d, Config(), "cpu"))
        self.assertEqual([stem for stem, _, _ in out], ["a", "b"])
        self.assertEqual(load.call_count, 4)
        self.assertEqual(vote.call_count, 2)


class TestMask2FormerMembers(unittest.TestCase):
    def test_threshold_and_first_owner_wins(self):
        high = (0.99, rle(0, 30, 0, 30))
        overlap = (0.95, rle(0, 30, 20, 50))
        low = (0.5, rle(40, 60, 40, 60))
        out = mask2former.member_masks([high, overlap, low], threshold=0.9)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(mask_util.area(out[0])), 900)
        self.assertEqual(int(mask_util.area(out[1])), 600)  # overlap loses its 300 owned pixels

    def test_nothing_above_threshold_keeps_the_best_query(self):
        out = mask2former.member_masks([(0.6, rle(0, 30, 0, 30)), (0.5, rle(40, 60, 40, 60))])
        self.assertEqual(len(out), 1)
        self.assertEqual(int(mask_util.area(out[0])), 900)

    def test_small_remainder_is_dropped(self):
        out = mask2former.member_masks([(0.99, rle(0, 30, 0, 30)), (0.98, rle(0, 31, 0, 30))])
        self.assertEqual(len(out), 1)  # the second keeps only 30 pixels, under the 250 px floor

    def test_no_queries(self):
        self.assertEqual(mask2former.member_masks([]), [])


if __name__ == "__main__":
    unittest.main()
