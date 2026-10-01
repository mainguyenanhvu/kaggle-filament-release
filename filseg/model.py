"""The two networks of the pipeline: a Mask R-CNN filament detector and a seed-box crop U-Net refiner.

Both come from torchvision (BSD-3-Clause). `pretrained=True` starts from the COCO detector and the
ImageNet ResNet-18 encoder, as in training; inference builds the bare architectures and loads our
checkpoints, so it downloads nothing.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))


def build_detector(
    img_size: int = 1024,
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    pretrained: bool = True,
) -> nn.Module:
    """Mask R-CNN ResNet-50-FPN v2 with two classes (background, filament), small anchors and a
    28x28 mask RoI pool. Images are resized to `img_size` on both sides."""
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=img_size,
        max_size=img_size,
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=ANCHOR_SIZES, aspect_ratios=(tuple(anchor_ratios),) * len(ANCHOR_SIZES)
    )
    n_anchor = len(anchor_ratios)
    if n_anchor != 3:
        # Keep the two pretrained RPN convolutions; only the 1x1 predictors change shape.
        c = model.rpn.head.cls_logits.in_channels
        model.rpn.head.cls_logits = nn.Conv2d(c, n_anchor, 1)
        model.rpn.head.bbox_pred = nn.Conv2d(c, 4 * n_anchor, 1)
        for layer in (model.rpn.head.cls_logits, model.rpn.head.bbox_pred):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return model


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CropUNetSeedbox(nn.Module):
    """U-Net with a ResNet-18 encoder over a square crop around one detection. The three input
    channels are the grey image, its CLAHE, and the detection's bounding box drawn as a filled
    rectangle (the "seed box"); the output is one logit map for the filament inside the box."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))


def load_detector(
    path, device, img_size: int = 1024, anchor_ratios=(0.5, 1.0, 2.0)
) -> nn.Module:
    model = build_detector(img_size, anchor_ratios, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.to(device).eval()


def load_refiner(path, device) -> nn.Module:
    model = CropUNetSeedbox(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.to(device).eval()
