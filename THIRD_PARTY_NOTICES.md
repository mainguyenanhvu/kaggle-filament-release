# Third-party notices

This repository's own code is released under the Apache License 2.0 (see `LICENSE`). It depends on
the third-party software and pretrained weights listed below. None is redistributed here as published;
our fine-tuned checkpoints in `weights/` derive from the two torchvision weights below.
Licence statements were checked on 2026-10-04 against the installed package metadata and the URLs
given.

## Pretrained weights

Both training scripts initialise from weights published by torchvision. The weights are downloaded
from `download.pytorch.org` at training time through `torch.hub` and are not redistributed in this
repository. Inference (`scripts/predict.py`, `scripts/predict_vote.py`) builds every network with
`pretrained=False` or from a saved config and loads only our fine-tuned checkpoints, so it downloads nothing.

| weights | file (sha256 prefix in name) | trained on | how it is used |
|---|---|---|---|
| `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | <https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth> | COCO 2017; training recipe <https://github.com/pytorch/vision/pull/5773> | initialises the detector in `scripts/train_detector.py`; box and mask predictors are replaced with two-class heads |
| `ResNet18_Weights.IMAGENET1K_V1` | <https://download.pytorch.org/models/resnet18-f37072fd.pth> | ImageNet-1k (ILSVRC 2012) | initialises the encoder of the crop U-Net refiner in `scripts/train_refiner.py` |

torchvision states no separate licence for its pretrained weights. Its documentation reads: "The
pre-trained models provided in this library may have their own licenses or terms and conditions
derived from the dataset used for training. It is your responsibility to determine whether you have
permission to use the models for your use case."
(<https://pytorch.org/vision/0.28/models.html>; also `README.md` of torchvision v0.28.0.)

| `convnext_tiny.dinov3_lvd1689m` (timm, Hugging Face hub `timm/convnext_tiny.dinov3_lvd1689m`) | downloaded by `timm` at training time only | Meta's DINOv3 self-supervised pretraining (LVD-1689M) | initialises the ConvNeXt-tiny backbone of the cascade detector (`scripts/train_detector.py --cascade`); inference builds it with `pretrained=False`. Distributed by Meta under the "DINOv3 License" (see the model card); not redistributed here; a checkpoint fine-tuned from it is subject to that licence, so check it before redistributing `cascade_detector.pt` |

| `convnext_small.in12k_ft_in1k` (timm, Hugging Face hub `timm/convnext_small.in12k_ft_in1k`) | downloaded by `timm` at training time only | ImageNet-12k, fine-tuned on ImageNet-1k; model card licence `apache-2.0` (read 2026-11-04) | initialises the ConvNeXt-small voter (`--backbone`) |
| `facebook/mask2former-swin-small-coco-instance` (Hugging Face hub) | downloaded by `transformers` at training time only | COCO instance segmentation; model card licence field reads `other`, with no licence link (read 2026-11-04). The Mask2Former code it comes from is MIT (<https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE>) | initialises the Mask2Former voter in `scripts/train_mask2former.py` |

The cascade voter's checkpoint `cascade.pt` is a fine-tuned derivative of the DINOv3 weights, so it is
distributed under the DINOv3 License, not under this repository's Apache-2.0: the agreement is in
`LICENSE-DINOv3.md`, as its section 1.b.i requires, and publications using it must acknowledge it
(section 1.b.ii). The competition's Winner License excludes such a model from the open-source grant:
"In the event that input data or pretrained models with an incompatible license are used to generate
your winning solution, you do not need to grant an open source license in the preceding Section for
that data and/or model(s)." The same sentence covers the Mask2Former pretraining, whose licence field
names no terms.

The training datasets carry their own terms:

- **COCO.** "The annotations in this dataset along with this website belong to the COCO Consortium
  and are licensed under a Creative Commons Attribution 4.0 License." "The COCO Consortium does not
  own the copyright of the images. Use of the images must abide by the Flickr Terms of Use."
  (<https://cocodataset.org/#termsofuse>)
- **ImageNet.** The terms of access state: "Researcher shall use the Database only for non-commercial
  research and educational purposes." (<https://image-net.org/download.php>)

The fine-tuned checkpoints (`detector.pt`, `refiner.pt`) are derived from these weights and from
the competition training data, and are stored in `weights/` (see `README.md`). Users who intend
commercial use of those checkpoints should assess the dataset terms above; retraining from random
initialisation (`pretrained=False`) removes the dependency on the pretrained weights but not on the
competition data.

## Code ported into the cascade detector

`CascadeRoIHeads` in `filseg/model.py` subclasses and reuses torchvision's `RoIHeads`,
`fastrcnn_loss`, `maskrcnn_loss`, `maskrcnn_inference`, `BoxCoder`, `Matcher` and `FeaturePyramidNetwork`
(BSD-3-Clause, Copyright (c) Soumith Chintala 2016); no torchvision source is copied. The method is the
Cascade R-CNN of Cai and Vasconcelos (CVPR 2018, <https://arxiv.org/abs/1712.00726>); the mean-of-stages
score at inference follows the mmdetection convention (Apache-2.0), reimplemented here, not copied.

## Python packages (`requirements.txt`)

| package | version | licence | how it is used |
|---|---|---|---|
| torch | 2.13.0 | BSD-3-Clause (PyTorch `LICENSE`); bundled components under Apache-2.0, MIT, BSD and BSL-1.0 | tensors, training, inference |
| torchvision | 0.28.0 | BSD-3-Clause | Mask R-CNN and ResNet-18 definitions, RoI ops, pretrained-weight loader |
| timm | 1.0.30 | Apache-2.0 (package metadata) | ConvNeXt-tiny feature backbone of the cascade detector (`filseg/model.py`, `TimmFPN`) |
| transformers | 5.17.0 | Apache-2.0 (package metadata) | Mask2Former voter (`filseg/mask2former.py`) |
| numpy | 2.5.2 | BSD-3-Clause (bundled components: 0BSD, MIT, Zlib, CC0-1.0) | array processing |
| pycocotools | 2.0.11 | BSD-2-Clause (package metadata: FreeBSD) | RLE mask encoding and COCO annotation handling |
| opencv-python-headless | 5.0.0.93 | OpenCV: Apache-2.0; wheel packaging: MIT; bundled FFmpeg binaries: LGPL-2.1 | image I/O, CLAHE, morphology |
| pillow | 12.3.0 | MIT-CMU (HPND) | image I/O |
| pandas | 3.0.5 | BSD-3-Clause | tables and submission CSV |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based) | figures |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | progress bars |
| ipykernel | 7.4.0 | BSD-3-Clause | executes `notebooks/pipeline.ipynb` headless (optional) |
| nbclient | 0.11.0 | BSD-3-Clause | executes `notebooks/pipeline.ipynb` headless (optional) |

Each package is installed from PyPI by the user; none is vendored in this repository. The full
licence texts ship inside each installed package's `*.dist-info` directory.
