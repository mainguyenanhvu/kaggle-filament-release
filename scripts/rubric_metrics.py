"""Score predictions with every quantity of the committee rubric.

Prints (and optionally writes) PQ, SQ, RQ, TP/FP/FN over all annotator records, the IoU and Dice
distributions of overlapping pairs, and the one-to-many / many-to-one degree histograms; with
--plots it also draws the three figures of the organisers' self-evaluation notebook.

  python scripts/rubric_metrics.py --ann <train annotations.json> --pred <submission.csv|preds.jsonl> \
      [--heldout] [--out metrics.json] [--plots figures/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.io import heldout_stems, load_predictions, load_records  # noqa: E402
from filseg.metrics import Evaluation, evaluate  # noqa: E402


def plot(ev: Evaluation, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, x in (("iou", np.asarray(ev.pair_iou)), ("dice", ev.pair_dice)):
        fig, ax = plt.subplots(figsize=(4, 3))
        w = np.ones_like(x) / max(len(x), 1)
        ax.hist(
            x,
            bins=np.linspace(0, 1, 51),
            weights=w,
            color="navy",
            alpha=0.7,
            edgecolor="black",
        )
        if len(x):
            ax.axvline(
                x.mean(), color="crimson", ls="--", label=f"mean = {x.mean():.3f}"
            )
            ax.legend(fontsize=8)
        ax.set_xlabel(name.upper() if name == "iou" else "Dice")
        ax.set_ylabel("probability")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_distribution.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    for deg, n in ev.gt_degree.items():
        ax.bar(
            deg + 0.5,
            n,
            width=0.8,
            color="crimson" if deg == 0 else "forestgreen",
            edgecolor="black",
        )
    for deg, n in ev.pred_degree.items():
        ax.bar(
            -(deg + 0.5),
            n,
            width=0.8,
            color="darkorange" if deg == 0 else "steelblue",
            edgecolor="black",
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        "predictions per GT filament (right)  |  GT filaments per prediction (left)"
    )
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "one_to_many_many_to_one.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", required=True, help="organiser COCO annotation file")
    ap.add_argument(
        "--pred", required=True, help="submission CSV or JSON lines predictions"
    )
    ap.add_argument(
        "--heldout",
        action="store_true",
        help="score only the 60-image validation split",
    )
    ap.add_argument("--out", type=Path, help="write the summary as JSON")
    ap.add_argument("--plots", type=Path, help="directory for the distribution figures")
    a = ap.parse_args()

    stems = heldout_stems(a.ann) if a.heldout else None
    records = load_records(a.ann, stems)
    ev = evaluate(records, load_predictions(a.pred))
    summary = ev.summary()
    summary["n_images"] = len(records)
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")
    if a.plots:
        plot(ev, a.plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
