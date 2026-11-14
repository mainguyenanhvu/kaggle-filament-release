"""Apply the guarded CPR strip refiner to a submission CSV (the second pass of CPRVOTE_G).

  python scripts/refine_cpr.py --images data/.../test/test_images --csv outputs/vote.csv \
      --weights weights/cpr.pt --out outputs/cprvote_G.csv

Every mask of `--csv` is refined on its own H-alpha image (`filseg.cpr`); rows keep their count and
order per image, images are written in sorted stem order. Run with OMP_NUM_THREADS=1 to match the
recorded build.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.cpr import load_strip_net, refine_image  # noqa: E402

cv2.setNumThreads(0)


def read_csv_rles(path: Path) -> dict[str, list[dict]]:
    """Image stem -> RLEs from a submission CSV, rows in file order.

    Parameters
    ----------
    path : Path
        Submission CSV with header ``filament_id,segmentation_rle``.

    Returns
    -------
    dict[str, list[dict]]
        COCO run-length masks (2048 x 2048) per image stem.
    """
    csv.field_size_limit(1 << 30)
    out: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        rows = csv.reader(f)
        if next(rows) != ["filament_id", "segmentation_rle"]:
            sys.exit(f"{path}: not a submission CSV")
        for fid, rle in rows:
            out.setdefault(fid.rsplit("_", 1)[0], []).append({"size": [2048, 2048], "counts": rle.encode()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path, help="submission CSV whose masks are refined")
    ap.add_argument("--weights", required=True, type=Path, help="weights/cpr.pt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_strip_net(args.weights, device)
    ref = read_csv_rles(args.csv)
    stems = sorted(ref)[: args.limit] if args.limit else sorted(ref)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    started = time.perf_counter()
    rows = 0
    with open(partial, "w", newline="", encoding="utf-8") as fh:
        fh.write("filament_id,segmentation_rle\n")
        for k, stem in enumerate(stems, 1):
            img = cv2.imread(str(args.images / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                sys.exit(f"missing image {stem}.jpeg in {args.images}")
            for n, r in enumerate(refine_image(net, img, ref[stem], device), 1):
                c = r["counts"]
                fh.write(f"{stem}_{n},{c.decode('ascii') if isinstance(c, bytes) else c}\n")
                rows += 1
            if k % 25 == 0:
                print(f"[{k}/{len(stems)}] {stem}", flush=True)
    partial.replace(args.out)
    md5 = hashlib.md5(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}: {rows} rows, {len(stems)} images, md5 {md5}, {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
