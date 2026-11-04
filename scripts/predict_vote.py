"""Run the voted pipeline on a folder of images and write a submission CSV.

  python scripts/predict_vote.py --images data/.../test/test_images --weights weights \
      --out outputs/submission.csv

`weights/vote.json` names the detectors (see `filseg.ensemble`). `--heldout-of <annotations.json>`
predicts only the 60-image validation split, for `scripts/rubric_metrics.py`. Per-image wall time goes
to `<out>.timing.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.ensemble import counts_of, vote_all  # noqa: E402
from filseg.infer import Config, submission_rows  # noqa: E402
from filseg.io import heldout_stems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, help="default: <weights>/vote.json")
    ap.add_argument("--out", required=True, type=Path, help=".csv (submission) or .jsonl")
    ap.add_argument("--heldout-of", type=Path, help="annotation file: its 60-image split only")
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    paths = sorted(args.images.glob("*.jpeg"))
    if args.heldout_of:
        keep = heldout_stems(args.heldout_of)
        paths = [p for p in paths if p.stem in keep]
    paths = paths[: args.limit] if args.limit else paths
    if not paths:
        sys.exit(f"no images in {args.images}")
    manifest = args.manifest or args.weights / "vote.json"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    seconds = []
    with open(partial, "w", newline="", encoding="utf-8") as f:
        if args.out.suffix == ".csv":
            f.write("filament_id,segmentation_rle\n")
        for stem, masks, dt in vote_all(paths, manifest, args.weights, Config(), device):
            seconds.append(dt)
            if args.out.suffix == ".csv":
                decoded = [mask_util.decode(m).astype(bool) for m in masks]
                f.writelines(r + "\n" for r in submission_rows(stem, decoded, (2048, 2048)))
            else:
                f.write(json.dumps({"stem": stem, "masks": counts_of(masks)}) + "\n")
    partial.replace(args.out)
    timing = {
        "images": len(seconds),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "total_s": round(sum(seconds), 2),
        "mean_s": round(statistics.mean(seconds), 3),
        "median_s": round(statistics.median(seconds), 3),
        "first_image_s": round(seconds[0], 3),
        "p95_s": round(float(np.percentile(seconds, 95)), 3),
    }
    args.out.with_name(args.out.name + ".timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    print(f"wrote {args.out}; timing {timing}", flush=True)


if __name__ == "__main__":
    main()
