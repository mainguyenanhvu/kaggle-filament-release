"""Run the pipeline over a folder of H-alpha images and write a submission CSV or JSON lines.

  python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
      --detector weights/detector.pt --refiner weights/refiner.pt --out submission.csv

  # the 60-image validation split, scored afterwards with scripts/rubric_metrics.py --heldout
  python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images \
      --heldout-of <train annotations.json> --detector ... --refiner ... --out heldout.jsonl

`--no-tta` disables the translate TTA (about 3x faster). The run prints per-image wall time and
writes a timing summary next to the output (`<out>.timing.json`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.infer import Config, encode, predict_all, submission_rows  # noqa: E402
from filseg.io import heldout_stems  # noqa: E402
from filseg.model import load_detector, load_refiner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--images", required=True, type=Path, help="folder of *.jpeg images"
    )
    ap.add_argument("--detector", required=True, type=Path)
    ap.add_argument("--refiner", required=True, type=Path)
    ap.add_argument(
        "--out", required=True, type=Path, help=".csv (submission) or .jsonl"
    )
    ap.add_argument(
        "--heldout-of",
        type=Path,
        help="annotation file: predict only its 60-image validation split",
    )
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--score-thresh", type=float, default=Config.score_thresh)
    ap.add_argument("--refine-thresh", type=float, default=Config.refine_thresh)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    args = ap.parse_args()

    cfg = Config(
        score_thresh=args.score_thresh,
        refine_thresh=args.refine_thresh,
        tta_shift=0 if args.no_tta else Config.tta_shift,
    )
    device = torch.device(args.device)
    paths = sorted(args.images.glob("*.jpeg"))
    if args.heldout_of:
        keep = heldout_stems(args.heldout_of)
        paths = [p for p in paths if p.stem in keep]
    paths = paths[: args.limit] if args.limit else paths
    if not paths:
        sys.exit(f"no images in {args.images}")

    detector = load_detector(args.detector, device)
    refiner = load_refiner(args.refiner, device)
    print(f"{len(paths)} images, device {device}, {cfg}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(
        args.out.name + ".partial"
    )  # a killed run leaves no valid-looking file
    seconds = []
    with open(partial, "w", newline="", encoding="utf-8") as f:
        if args.out.suffix == ".csv":
            f.write("filament_id,segmentation_rle\n")
        for stem, masks, shape, dt in predict_all(paths, detector, refiner, cfg, device):
            seconds.append(dt)
            if args.out.suffix == ".csv":
                f.writelines(
                    r + "\n"
                    for r in submission_rows(stem, masks, shape)
                )
            else:
                f.write(
                    json.dumps({"stem": stem, "masks": [encode(m) for m in masks]})
                    + "\n"
                )
    partial.replace(args.out)

    timing = {
        "images": len(seconds),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tta": not args.no_tta,
        "total_s": round(sum(seconds), 2),
        "mean_s": round(statistics.mean(seconds), 3),
        "median_s": round(statistics.median(seconds), 3),
        "first_image_s": round(seconds[0], 3),
    }
    args.out.with_name(args.out.name + ".timing.json").write_text(
        json.dumps(timing, indent=1) + "\n"
    )
    print(f"wrote {args.out}; timing {timing}", flush=True)


if __name__ == "__main__":
    main()
