# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

Instance segmentation of solar filaments in full-disk H-alpha images: a Mask R-CNN detector proposes
filaments, and a seed-box crop U-Net refines each proposal's mask. This repository holds the code, the
configuration and the weights of the model we submit for evaluation (`weights/`, submission v36); it
contains no competition images or annotations.

Training, inference and the rubric scorer are all here, so the submitted result can be rebuilt from the
challenge data alone.

## Layout

| path | content |
|---|---|
| `filseg/model.py` | Mask R-CNN ResNet-50-FPN v2 detector and the seed-box crop U-Net refiner (ResNet-18 encoder) |
| `filseg/data.py` | the train/held-out split, the two record-deduplication rules, the detector's augmentation, the refiner's crop cache |
| `scripts/train_detector.py` | trains the detector: 18 epochs, SGD, the recipe of the submitted run |
| `scripts/train_refiner.py` | caches the seed-box crops and trains the refiner: 8 epochs, AdamW, BCE + soft Dice |
| `filseg/infer.py` | translate TTA, mask NMS, score threshold and pixel ownership, crop refinement, submission rows |
| `scripts/predict.py` | images -> submission CSV or JSON lines, with per-image wall time |
| `notebooks/pipeline.ipynb` | the whole pipeline: test images -> CSV, then the rubric metrics on the validation split |
| `filseg/metrics.py` | Panoptic Quality over every annotator record, IoU/Dice distributions, one-to-many and many-to-one counts |
| `filseg/io.py` | readers for the organiser COCO annotations, the submission CSV and JSON-lines predictions; the 60-image validation split |
| `scripts/rubric_metrics.py` | command-line scorer that prints every rubric quantity and draws the three distribution figures |
| `weights/` | the two checkpoints of the evaluated model (Git LFS), with `SHA256SUMS` and `manifest.json` |
| `tests/` | synthetic unit tests (no data, no weights, no network): `python -m unittest discover tests` |

## Data

Download the challenge data from Kaggle (`kaggle competitions download -c filament-segmentation-2026`)
and unpack it under `data/`, which git ignores. Training uses only the challenge training images and
annotations (optionally with unlabelled GONG frames); inference uses only the H-alpha test images.

## Training

The detector first, then the refiner; the two are independent and read the same annotation file.

```
python scripts/train_detector.py \
    --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images --out checkpoints/detector

python scripts/train_refiner.py \
    --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images --out checkpoints/refiner
```

The defaults are the recipe of the submitted run; `--help` lists every knob. The detector is
`checkpoints/detector/epoch_17.pt` (the last of 18 epochs, about 30 minutes per 18 epochs on one
RTX PRO 6000) and the refiner is `checkpoints/refiner/best_refiner_seedbox.pt` (the epoch with the
best held-out Dice; it also writes `refiner_history.csv`). Both trainers hold out the same 60
observations that `scripts/rubric_metrics.py --heldout` scores, so no held-out frame is ever trained on.

Training is not bit-reproducible: the augmentation draws and the GPU kernels are not pinned to one
sequence, and five runs of this recipe span a held-out PQ of 0.4197 +- 0.0085 (mean +- sd). What is
reproducible is the data path: the split, the two deduplication rules, the augmentation and the crop
cache are deterministic in their seeds, and the submitted weights are published so that the reported
numbers can be reproduced exactly (below).

## Inference

The two checkpoints are stored in this repository under `weights/` with Git LFS, and the notebook loads
them from there. Clone with `git lfs install` done once, or run `git lfs pull` after cloning; a clone
without LFS holds small pointer files, which the notebook reports. A GitHub "Download ZIP" may hold
pointers too. Each file is listed in `weights/SHA256SUMS` and `weights/manifest.json` records which
training checkpoint it came from. Verify them with `cd weights && sha256sum -c SHA256SUMS`:

| file | sha256 | bytes | parameters |
|---|---|---|---|
| `detector.pt` | `56d9005dfcaad8c3c00986fb0ce80a2fae561bd817e067a87a59ab7b36264c4d` | 183,913,877 | 45,941,792 |
| `refiner.pt` | `4dccebbd7de0f41e5e3cb3d00dd4c7fd0960c326c14a0f44fa43adf349d98e8f` | 57,048,983 | 14,248,157 |

Each file is `{"model": state_dict, "epoch": int}`: the training checkpoint without its optimiser and
scheduler state, bit-identical in every weight tensor. Then

```
python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --detector weights/detector.pt --refiner weights/refiner.pt --out outputs/submission.csv
```

`--no-tta` drops the translate TTA (about three times faster). The pipeline also runs on CPU
(`--device cpu`), at about 2.2 s per image without TTA on eight threads.

`python scripts/vote.py --base base.csv --voter allrec:R50=allrec.csv --voter ridge:R50=ridge.csv --voter dapt:R50=dapt.csv --voter m2f=m2f.csv --voter cnxs=cnxs.csv --out consensus.csv`
Class-balanced consensus gives each linked class one vote and the base a separate vote (IoU > 0.4, pixel support >= 0.5).

`notebooks/pipeline.ipynb` runs the evaluated model end to end. It reads the challenge data from `data/`
(or `FILSEG_DATA=<dir holding test/ and train/>`). Without that folder it writes one synthetic full-disk
image, so a fresh clone still exercises every inference step; the rubric section then reports that it was
skipped. `FILSEG_LIMIT=N` keeps the first N images of each split for a quick check:

```
pip install -r requirements.txt
jupyter execute notebooks/pipeline.ipynb
```

The class-balanced vote above is the development route to our best public score, not the evaluated model:
its members are not all in this repository.

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

## Reproducing the reported numbers

The two commands below rebuild the submitted test file and the held-out table of the report.

```
# the submitted test predictions: MD5 7d6699df05b519b690da3b341e9739f2, 1383 rows, 180 images
python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --detector weights/detector.pt --refiner weights/refiner.pt --out outputs/test.csv

# the held-out metrics: PQ 0.4330 over 60 observations / 100 annotator records
python scripts/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images \
    --heldout-of data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --detector weights/detector.pt --refiner weights/refiner.pt --out outputs/heldout.jsonl
python scripts/rubric_metrics.py --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --pred outputs/heldout.jsonl --heldout --bootstrap 5000 --out metrics.json
```

Measured on one RTX PRO 6000: the test file is byte-identical to the submitted one, and the held-out
split gives PQ 0.4330 (SQ 0.6724, RQ 0.6439; TP/FP/FN 510/283/281; 95% interval 0.397 to 0.467) in
231 s of model time. The report quotes 0.43166 (508/284/283) for the same weights and operating point,
measured during development by replaying cached detections through the same post-processing rather than
by a fresh forward pass; the two agree mask for mask (488 against 487 masks, median best IoU 1.000, 91%
of masks above 0.99) and the 0.0013 difference is a sixth of the 0.0085 run-to-run standard deviation of
the recipe. On CPU the same command gives 0.4324.

## Licence

Apache-2.0 (see `LICENSE`). Every dependency in `requirements.txt` is under a permissive OSI licence. The
checkpoints in `weights/` derive from COCO and ImageNet pretraining; see `THIRD_PARTY_NOTICES.md`.
