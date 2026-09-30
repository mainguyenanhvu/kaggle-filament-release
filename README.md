# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

Instance segmentation of solar filaments in full-disk H-alpha images: a Mask R-CNN detector proposes
filaments, and a seed-box crop U-Net refines each proposal's mask. This repository holds the code only;
it contains no competition images, annotations or trained weights.

Status: work in progress. `filseg.metrics` and `scripts/rubric_metrics.py` are complete; the training
and inference pipeline and the end-to-end notebook are being ported here.

## Layout

| path | content |
|---|---|
| `filseg/metrics.py` | Panoptic Quality over every annotator record, IoU/Dice distributions, one-to-many and many-to-one counts |
| `filseg/io.py` | readers for the organiser COCO annotations, the submission CSV and JSON-lines predictions; the 60-image validation split |
| `scripts/rubric_metrics.py` | command-line scorer that prints every rubric quantity and draws the three distribution figures |
| `tests/` | synthetic unit tests (no data needed): `python -m unittest discover tests` |

## Data

Download the challenge data from Kaggle (`kaggle competitions download -c filament-segmentation-2026`)
and unpack it under `data/`, which git ignores. Training uses only the challenge training images and
annotations (optionally with unlabelled GONG frames); inference uses only the H-alpha test images.

## Scoring

```
pip install -r requirements.txt
python scripts/rubric_metrics.py --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --pred predictions.jsonl --heldout --out metrics.json --plots figures/
```

The scorer follows the organisers' self-evaluation notebook: each (annotator, image) record is scored
against the image's predictions, a pair is a true positive when IoU > 0.5, and counts are pooled over all
records, so a prediction is a false positive once per annotator who did not draw it.

## Licence

Apache-2.0 (see `LICENSE`). Every dependency in `requirements.txt` is under a permissive OSI licence.
