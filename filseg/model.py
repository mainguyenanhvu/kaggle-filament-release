"""The two networks of the pipeline: a Mask R-CNN filament detector and a seed-box crop U-Net refiner.

Both come from torchvision (BSD-3-Clause). `pretrained=True` starts from the COCO detector and the
ImageNet ResNet-18 encoder, as in training; inference builds the bare architectures and loads our
checkpoints, so it downloads nothing.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
)
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))


class TimmFPN(nn.Module):
    """A timm `features_only` backbone (stages 0-3) under a fresh 4-level FPN with batch norm and a
    max-pool extra level, in the dict-of-features form torchvision's detector expects. Module names
    (`body.m`, `fpn`) are the checkpoint keys of the ConvNeXt detectors."""

    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        import timm  # imported here: only the ConvNeXt (cascade) detector needs it

        self.body = nn.Module()
        self.body.m = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.fpn = FeaturePyramidNetwork(
            self.body.m.feature_info.channels(),
            256,
            extra_blocks=LastLevelMaxPool(),
            norm_layer=nn.BatchNorm2d,
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor):
        feats = OrderedDict((str(i), f) for i, f in enumerate(self.body.m(x)))
        return self.fpn(feats)


# Cascade R-CNN (Cai & Vasconcelos, CVPR 2018): three box stages.
CASCADE_FG_IOUS = (0.5, 0.6, 0.7)
CASCADE_CODER_W = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
CASCADE_STAGE_W = (1.0, 0.5, 0.25)
CONVNEXT = "convnext_tiny.dinov3_lvd1689m"


class CascadeRoIHeads(RoIHeads):
    """Replaces the single box stage of a torchvision Mask R-CNN by three.

    Stage k matches proposals at IoU 0.5 / 0.6 / 0.7, codes boxes with weights (10,10,5,5) /
    (20,20,10,10) / (30,30,15,15) and weighs its loss 1 / 0.5 / 0.25; the next stage's proposals are
    the previous stage's decoded boxes. The mask head trains on the last stage's positives. At
    inference the final score is the mean of the three stages' softmax outputs, the final boxes are
    stage 3's regression, and they feed the unchanged mask head. The output is torchvision's dict
    (boxes, labels, scores, masks). Stage 0 keeps the stock `box_head` / `box_predictor` names;
    stages 1 and 2 are `extra_box_heads[0..1]` / `extra_box_predictors[0..1]`.
    """

    def __init__(self, base: RoIHeads):
        super().__init__(
            base.box_roi_pool,
            base.box_head,
            base.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=base.fg_bg_sampler.batch_size_per_image,
            positive_fraction=base.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=CASCADE_CODER_W[-1],
            score_thresh=base.score_thresh,
            nms_thresh=base.nms_thresh,
            detections_per_img=base.detections_per_img,
            mask_roi_pool=base.mask_roi_pool,
            mask_head=base.mask_head,
            mask_predictor=base.mask_predictor,
        )
        self.extra_box_heads = nn.ModuleList([copy.deepcopy(base.box_head) for _ in range(2)])
        self.extra_box_predictors = nn.ModuleList(
            [copy.deepcopy(base.box_predictor) for _ in range(2)]
        )
        self.coders = [det_utils.BoxCoder(w) for w in CASCADE_CODER_W]
        self.matchers = [
            det_utils.Matcher(t, t, allow_low_quality_matches=False) for t in CASCADE_FG_IOUS
        ]
        self.box_coder = self.coders[-1]  # used by postprocess_detections: final-stage boxes
        self.rescale_stage_regressors()

    @torch.no_grad()
    def rescale_stage_regressors(self):
        """Stage k's bbox_pred x (w_k / w_0): at step 0 every stage decodes the stage-0 offsets."""
        for k, pred in enumerate(self.extra_box_predictors, start=1):
            s = CASCADE_CODER_W[k][0] / CASCADE_CODER_W[0][0]
            pred.bbox_pred.weight.mul_(s)
            pred.bbox_pred.bias.mul_(s)

    def stage_modules(self, k):
        if k == 0:
            return self.box_head, self.box_predictor
        return self.extra_box_heads[k - 1], self.extra_box_predictors[k - 1]

    def run_stage(self, k, features, props, image_shapes):
        head, pred = self.stage_modules(k)
        logits, reg = pred(head(self.box_roi_pool(features, props, image_shapes)))
        return logits.float(), reg.float()

    def refine(self, k, reg, props, image_shapes):
        """Stage k's foreground regression applied to its RoIs: the next stage's proposals
        (detached, clipped to the image, at least 1 px wide and high)."""
        n = [len(p) for p in props]
        boxes = self.coders[k].decode(reg.detach(), props)[:, 1]
        out = []
        for b, shape in zip(boxes.split(n, 0), image_shapes):
            b = box_ops.clip_boxes_to_image(b, shape)
            x1, y1 = b[:, 0], b[:, 1]
            out.append(
                torch.stack(
                    [x1, y1, torch.maximum(b[:, 2], x1 + 1.0), torch.maximum(b[:, 3], y1 + 1.0)], 1
                )
            )
        return out

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            return self._forward_train(features, proposals, image_shapes, targets)
        return self._forward_eval(features, proposals, image_shapes)

    def _forward_train(self, features, proposals, image_shapes, targets):
        self.check_targets(targets)
        dtype, device = proposals[0].dtype, proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        losses = {}
        props = [p.detach() for p in proposals]
        for k in range(3):
            n_in = [len(p) for p in props]
            cand = self.add_gt_proposals(props, gt_boxes)
            self.proposal_matcher = self.matchers[k]
            matched, labels = self.assign_targets_to_proposals(cand, gt_boxes, gt_labels)
            sampled = self.subsample(labels)
            s_props, s_labels, s_matched, s_isgt, mgt = [], [], [], [], []
            for i in range(len(cand)):
                idx = sampled[i]
                s_props.append(cand[i][idx])
                s_labels.append(labels[i][idx])
                s_matched.append(matched[i][idx])
                s_isgt.append(idx >= n_in[i])
                g = (
                    gt_boxes[i]
                    if gt_boxes[i].numel()
                    else torch.zeros((1, 4), dtype=dtype, device=device)
                )
                mgt.append(g[s_matched[-1]])
            reg_t = self.coders[k].encode(mgt, s_props)
            logits, reg = self.run_stage(k, features, s_props, image_shapes)
            lc, lb = fastrcnn_loss(logits, reg, s_labels, reg_t)
            losses[f"loss_classifier_s{k}"] = lc * CASCADE_STAGE_W[k]
            losses[f"loss_box_reg_s{k}"] = lb * CASCADE_STAGE_W[k]
            if k < 2:
                nxt = self.refine(k, reg, s_props, image_shapes)
                # drop the refined GT copies; the next stage re-appends the GT
                props = [b[~g] for b, g in zip(nxt, s_isgt)]
        mask_props, pos_matched = [], []
        for i in range(len(s_props)):
            pos = torch.where(s_labels[i] > 0)[0]
            mask_props.append(s_props[i][pos])
            pos_matched.append(s_matched[i][pos])
        mask_logits = self.mask_predictor(
            self.mask_head(self.mask_roi_pool(features, mask_props, image_shapes))
        )
        gt_masks = [t["masks"] for t in targets]
        losses["loss_mask"] = maskrcnn_loss(
            mask_logits, mask_props, gt_masks, gt_labels, pos_matched
        )
        return [], losses

    def _forward_eval(self, features, proposals, image_shapes):
        props = proposals
        probs = []
        for k in range(3):
            logits, reg = self.run_stage(k, features, props, image_shapes)
            probs.append(F.softmax(logits, -1))
            if k < 2:
                props = self.refine(k, reg, props, image_shapes)
        score = torch.stack(probs).mean(0)
        boxes, scores, labels = self.postprocess_detections(
            torch.log(score.clamp_min(1e-12)), reg, props, image_shapes
        )
        result = [{"boxes": b, "labels": l, "scores": s} for b, s, l in zip(boxes, scores, labels)]
        mf = self.mask_head(
            self.mask_roi_pool(features, [r["boxes"] for r in result], image_shapes)
        )
        masks = maskrcnn_inference(self.mask_predictor(mf), [r["labels"] for r in result])
        for m, r in zip(masks, result):
            r["masks"] = m
        return result, {}


def build_detector(
    img_size: int = 1024,
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    pretrained: bool = True,
    backbone: str = "resnet50",
    cascade: bool = False,
) -> nn.Module:
    """Mask R-CNN ResNet-50-FPN v2 with two classes (background, filament), small anchors and a
    28x28 mask RoI pool. Images are resized to `img_size` on both sides.

    `backbone` other than "resnet50" is a timm model name under a fresh FPN (TimmFPN); `cascade`
    swaps the box stage for CascadeRoIHeads. The record cascade detector is
    backbone=CONVNEXT, cascade=True."""
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
    if backbone != "resnet50":
        model.backbone = TimmFPN(backbone, pretrained)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    if cascade:
        model.roi_heads = CascadeRoIHeads(model.roi_heads)
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
    path,
    device,
    img_size: int = 1024,
    anchor_ratios=(0.5, 1.0, 2.0),
    backbone: str = "resnet50",
    cascade: bool = False,
) -> nn.Module:
    model = build_detector(
        img_size, anchor_ratios, pretrained=False, backbone=backbone, cascade=cascade
    )
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.to(device).eval()


def load_refiner(path, device) -> nn.Module:
    model = CropUNetSeedbox(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.to(device).eval()
