"""Write the cached candidates of a detector (the input of `scripts/vote.py --extra`).

  python scripts/predict_cands.py --images <train_images> --heldout-of <train annotations.json> \
      --detector checkpoints/cascade/epoch_02.pt --cascade --refiner weights/refiner.pt --out cands.pkl

Per image: every detection scoring >= 0.30 after translate TTA and mask NMS, owned in score order,
with the refiner's probability map and crop bounds (see `filseg.cascade`). The pickle is our own
output; never load one from elsewhere.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.cascade import candidates  # noqa: E402
from filseg.infer import Config  # noqa: E402
from filseg.io import heldout_stems  # noqa: E402
from filseg.model import CONVNEXT, load_detector, load_refiner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--detector", required=True, type=Path)
    ap.add_argument("--refiner", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cascade", action="store_true", help="the ConvNeXt cascade detector")
    ap.add_argument("--heldout-of", type=Path, help="annotation file: its 60-image split only")
    ap.add_argument("--limit", type=int, help="first N images only (after sorting)")
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
    detector = (
        load_detector(args.detector, device, backbone=CONVNEXT, cascade=True)
        if args.cascade
        else load_detector(args.detector, device)
    )
    refiner = load_refiner(args.refiner, device)
    out = candidates(paths, detector, refiner, Config(), device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out}: {len(out)} images")


if __name__ == "__main__":
    main()
