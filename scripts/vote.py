"""Reshape base CSV instances by class-balanced pixel consensus.

Each voter contributes its highest-IoU mask if IoU > 0.4 to the base object.
Linked voters sharing a class share one vote; the base has a separate vote.
Pixels with support >= 0.5 survive in greedy disjoint ownership order.
Shapes under 250 pixels fall back to the base; instance counts are preserved.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.io import load_predictions, write_predictions_csv  # noqa: E402
from filseg.cascade import member_masks  # noqa: E402
from filseg.vote import cb_vote  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--voter", action="append", default=[], metavar="NAME[:CLASS]=PATH.csv")
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=CANDS.pkl",
        help="a cached-candidates voter in its own class, e.g. cascadecpu=cands.pkl (written by "
        "scripts/predict_cands.py); read at score 0.80, refiner 0.40, 250 px",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    voters, classes = {}, {}
    for spec in args.voter:
        label, sep, path = spec.partition("=")
        name, colon, cls = label.partition(":")
        if not sep or not name or not path or (colon and not cls):
            ap.error(f"invalid voter specification: {spec}")
        if name in voters:
            ap.error(f"duplicate voter name: {name}")
        voters[name] = load_predictions(path)
        if colon:
            classes[name] = cls
    for spec in args.extra:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in voters:
            ap.error(f"invalid or duplicate --extra specification: {spec}")
        with open(path, "rb") as f:
            cands = pickle.load(f)  # our own output; do not load pickles from elsewhere
        voters[name] = {stem: member_masks(entry) for stem, entry in cands.items()}
    base = load_predictions(args.base)
    result = {
        stem: cb_vote(rles, {name: v.get(stem, []) for name, v in voters.items()}, classes)
        for stem, rles in base.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(args.out, result)
    print(f"wrote {args.out}: {sum(len(r) for r in result.values())} instances")


if __name__ == "__main__":
    main()
