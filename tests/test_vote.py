"""Class-balanced consensus on synthetic masks only."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from filseg.io import load_predictions, write_predictions_csv
from filseg.vote import cb_vote


def rle(mask):
    return mask_util.encode(np.asarray(mask, np.uint8, order="F"))


def masks():
    base = np.zeros((40, 40), bool)
    base[5:25, 5:25] = True
    wider = base.copy()
    wider[5:25, 25:30] = True
    return base, wider


def test_no_voters():
    base, _ = masks()
    encoded = [rle(base)]
    assert cb_vote(encoded, {}) == encoded
    assert cb_vote([], {"a": encoded}) == []


def test_full_agreement():
    base, _ = masks()
    result = cb_vote([rle(base)], {"a": [rle(base)], "b": [rle(base)]})
    np.testing.assert_array_equal(mask_util.decode(result[0]), base)


def test_three_members_share_one_vote():
    base, wider = masks()
    voters = {name: [rle(wider)] for name in ("a", "b", "c")}
    voters["d"] = [rle(base)]
    classes = {"a": "shared", "b": "shared", "c": "shared"}
    balanced = cb_vote([rle(base)], voters, classes)
    individual = cb_vote([rle(base)], voters)
    np.testing.assert_array_equal(mask_util.decode(balanced[0]), base)
    np.testing.assert_array_equal(mask_util.decode(individual[0]), wider)


def test_only_linked_members_share_class_vote():
    base, wider = masks()
    unrelated = np.zeros_like(base)
    unrelated[30:35, 30:35] = True
    voters = {"a": [rle(wider)], "b": [], "c": [rle(unrelated)]}
    classes = {name: "shared" for name in voters}
    result = cb_vote([rle(base)], voters, classes)
    np.testing.assert_array_equal(mask_util.decode(result[0]), wider)


def test_small_consensus_falls_back():
    base, _ = masks()
    smaller = base.copy()
    smaller[15:25] = False
    voters = {name: [rle(smaller)] for name in ("a", "b", "c")}
    result = cb_vote([rle(base)], voters)
    np.testing.assert_array_equal(mask_util.decode(result[0]), base)


def test_greedy_ownership_preserves_instance_count():
    base, wider = masks()
    result = cb_vote([rle(base), rle(wider)], {"a": [rle(wider)]})
    assert len(result) == 2
    np.testing.assert_array_equal(mask_util.decode(result[0]), wider)
    assert not mask_util.decode(result[1]).any()


def test_cli_round_trip(tmp_path):
    base = np.zeros((2048, 2048), bool)
    base[5:25, 5:25] = True
    wider = base.copy()
    wider[5:25, 25:30] = True
    base_path, voter_path, out_path = [
        tmp_path / n for n in ("base.csv", "voter.csv", "out.csv")
    ]
    write_predictions_csv(base_path, {"synthetic_frame": [rle(base)]})
    write_predictions_csv(voter_path, {"synthetic_frame": [rle(wider)]})
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "vote.py"),
        "--base", str(base_path), "--out", str(out_path),
    ]
    for name in ("a", "b", "c"):
        cmd += ["--voter", f"{name}:shared={voter_path}"]
    cmd += ["--voter", f"d={base_path}"]
    subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"}, timeout=30,
    )
    result = load_predictions(out_path)
    assert list(result) == ["synthetic_frame"]
    assert len(result["synthetic_frame"]) == 1
    np.testing.assert_array_equal(mask_util.decode(result["synthetic_frame"][0]), base)
