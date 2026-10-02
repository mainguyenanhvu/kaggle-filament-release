# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

Instance segmentation of solar filaments in full-disk H-alpha images: a Mask R-CNN detector proposes
filaments, and a seed-box crop U-Net refines each proposal's mask. This repository holds the code only;
it contains no competition images, annotations or trained weights.

Status: work in progress. Inference (`filseg/infer.py`, `scripts/predict.py`, `notebooks/pipeline.ipynb`)
and the rubric scorer are complete; the training scripts are being ported here.

## Layout

| path | content |
|---|---|
| `filseg/metrics.py` | Panoptic Quality over every annotator record, IoU/Dice distributions, one-to-many and many-to-one counts |
| `filseg/model.py` | Mask R-CNN ResNet-50-FPN v2 detector and the seed-box crop U-Net refiner (ResNet-18 encoder) |
| `filseg/infer.py` | translate TTA, mask NMS, score threshold and pixel ownership, crop refinement, submission rows |
| `scripts/predict.py` | images -> submission CSV or JSON lines, with per-image wall time |
| `notebooks/pipeline.ipynb` | the whole pipeline: test images -> CSV, then the rubric metrics on the validation split |
| `filseg/io.py` | readers for the organiser COCO annotations, the submission CSV and JSON-lines predictions; the 60-image validation split |
| `scripts/rubric_metrics.py` | command-line scorer that prints every rubric quantity and draws the three distribution figures |
| `tests/` | synthetic unit tests (no data needed): `python -m unittest discover tests` |

## Data

Download the challenge data from Kaggle (`kaggle competitions download -c filament-segmentation-2026`)
and unpack it under `data/`, which git ignores. Training uses only the challenge training images and
annotations (optionally with unlabelled GONG frames); inference uses only the H-alpha test images.

## Inference

The two checkpoints (`weights/detector.pt`, `weights/refiner.pt`) are distributed separately from the
code. Then

```
python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --detector weights/detector.pt --refiner weights/refiner.pt --out outputs/submission.csv
```

`--no-tta` drops the translate TTA. The pipeline also runs on CPU (`--device cpu`).

## Scoring

```
pip install -r requirements.txt
python scripts/rubric_metrics.py --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --pred predictions.jsonl --heldout --bootstrap 5000 --out metrics.json --plots figures/
```

The scorer follows the organisers' self-evaluation notebook: each (annotator, image) record is scored
against the image's predictions, a pair is a true positive when IoU > 0.5, and counts are pooled over all
records, so a prediction is a false positive once per annotator who did not draw it.

Beyond PQ, SQ, RQ and TP/FP/FN it reports the IoU and Dice quantiles of overlapping and of matched pairs,
the one-to-many and many-to-one counts at IoU > 0 and at the match threshold IoU > 0.5, PQ by the number of
annotator records per image, and (`--bootstrap N`) 95% intervals from resampling images with all their records.

## Licence

Apache-2.0 (see `LICENSE`). Every dependency in `requirements.txt` is under a permissive OSI licence.
