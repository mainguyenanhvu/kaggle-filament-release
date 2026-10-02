"""Train the seed-box crop U-Net refiner — the second of the two networks of the submitted pipeline.

  python scripts/train_refiner.py \
      --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
      --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images \
      --out checkpoints/refiner

Recipe: a U-Net with an ImageNet ResNet-18 encoder over a 256x256 square crop around one instance,
1.8 x its longer box side. The three input channels are the grey frame, its CLAHE and the seed box
drawn as a filled rectangle; the box is jittered (scale +-15%, centre +-10% of its size) so the
network trains on detector-like boxes rather than ground-truth ones. AdamW (lr 3e-4, weight decay
1e-4), cosine schedule, loss 0.5 BCE + 0.5 soft Dice, batch 16, 8 epochs, mixed precision on GPU.

The crops are cached to .npy memory maps first (one pass over the training frames), then the epoch
with the best held-out Dice at probability 0.65 is written to `best_refiner_seedbox.pt`. One record
per frame, the median-count one; the 60 held-out frames are the same ones the detector never sees.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.data import (  # noqa: E402
    CropDataset,
    build_crop_cache,
    dedup_median_count,
    load_coco,
    split_stems,
)
from filseg.model import CropUNetSeedbox  # noqa: E402

VAL_PROB = (
    0.65  # probability at which the validation Dice is measured (model selection only)
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument(
        "--out", required=True, type=Path, help="checkpoint and crop-cache directory"
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--context", type=float, default=1.8)
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--reuse-cache",
        action="store_true",
        help="keep an existing crop cache in --out instead of rebuilding it",
    )
    return ap.parse_args()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


@torch.inference_mode()
def validation_dice(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total, seen = 0.0, 0
    for images, masks in loader:
        masks = masks.to(device, non_blocking=True).bool()
        predicted = model(images.to(device, non_blocking=True)).sigmoid() >= VAL_PROB
        intersection = (predicted & masks).sum((1, 2, 3)).float()
        denominator = predicted.sum((1, 2, 3)) + masks.sum((1, 2, 3))
        total += ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item()
        seen += len(images)
    return total / max(seen, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint: Path,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict] = []
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = bce_sum = dice_sum = seen = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(images)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                dice = soft_dice_loss(logits, masks)
                loss = 0.5 * bce + 0.5 * dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = len(images)
            loss_sum += loss.item() * batch
            bce_sum += bce.item() * batch
            dice_sum += dice.item() * batch
            seen += batch

        val_dice = float("nan")
        if val_loader is not None:
            val_dice = validation_dice(model, val_loader, device)
            if val_dice > best:
                best = val_dice
                torch.save({"epoch": epoch, "model": model.state_dict()}, checkpoint)
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "bce": bce_sum / max(seen, 1),
            "dice_loss": dice_sum / max(seen, 1),
            "val_dice": val_dice,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} loss {row['loss']:.4f} bce {row['bce']:.4f} "
            f"dice loss {row['dice_loss']:.4f} val Dice {val_dice:.4f}",
            flush=True,
        )

    if val_loader is None:
        torch.save({"epoch": epochs, "model": model.state_dict()}, checkpoint)
    return history


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.ann)
    train_stems, val_stems = split_stems(coco, args.n_val, args.split_seed)
    chosen = dedup_median_count(coco)
    print(
        f"{len(train_stems)} train frames, {len(val_stems)} held out, device {device}",
        flush=True,
    )

    rng = random.Random(args.jitter_seed)
    cache = {}
    for prefix, stems in (("train", train_stems), ("val", val_stems)):
        images = args.out / f"{prefix}_crop_images3ch.npy"
        masks = args.out / f"{prefix}_crop_masks.npy"
        if args.reuse_cache and images.exists() and masks.exists():
            print(f"reusing {images}", flush=True)
        else:
            images, masks, n = build_crop_cache(
                coco,
                stems,
                chosen,
                args.images,
                args.out,
                prefix,
                rng,
                args.crop_size,
                args.context,
                args.min_side,
            )
            print(f"{prefix}: {n} crops", flush=True)
        cache[prefix] = (images, masks)

    train_loader = DataLoader(
        CropDataset(*cache["train"], augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CropDataset(*cache["val"], augment=False),
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = args.out / "best_refiner_seedbox.pt"
    history = train(
        CropUNetSeedbox(pretrained=True),
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
        checkpoint,
    )
    (args.out / "refiner_history.csv").write_text(
        "epoch,loss,bce,dice_loss,val_dice\n"
        + "".join(
            f"{r['epoch']},{r['loss']:.6f},{r['bce']:.6f},{r['dice_loss']:.6f},{r['val_dice']:.6f}\n"
            for r in history
        )
    )
    print(f"done; refiner = {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
