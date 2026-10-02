"""Train the Mask R-CNN filament detector — the first of the two networks of the submitted pipeline.

  python scripts/train_detector.py \
      --ann data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
      --images data/MAGFiLO_1.0_Kaggle_2026/train/train_images \
      --out checkpoints/detector

Recipe: COCO-pretrained Mask R-CNN ResNet-50-FPN v2, two classes, anchors (16, 32, 64, 128, 256) px
with three aspect ratios per level so the pretrained RPN head is kept, a 28x28 mask RoI pool, images
resized to 1024 px on both sides, SGD (lr 0.005, momentum 0.9, weight decay 1e-4) with mixed
precision, 18 epochs, lr x0.1 after epochs 12 and 16, batch 4. Augmentation: random 90-degree
rotation, horizontal flip and on-disk photometric jitter. One record per frame, the most complete one.

The run writes `epoch_NN.pt` (model, optimizer, scheduler, epoch) and keeps the last two; the
submitted detector is the last epoch, `epoch_17.pt`. Re-running resumes from the latest checkpoint.
Training is deterministic only in its data split: the augmentation and cuDNN kernels are not seeded
to one draw, and repeated runs of this recipe have a held-out PQ standard deviation of 0.0085
(measured over five draws, reported in the four-page report).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.data import (  # noqa: E402
    DetectorDataset,
    collate,
    dedup_most_complete,
    load_coco,
    split_stems,
)
from filseg.model import build_detector  # noqa: E402

MILESTONE_FRACTIONS = (0.7, 0.9)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--ann", required=True, type=Path, help="organisers' COCO training annotations"
    )
    ap.add_argument(
        "--images", required=True, type=Path, help="folder of training *.jpeg frames"
    )
    ap.add_argument("--out", required=True, type=Path, help="checkpoint directory")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers")
    ap.add_argument(
        "--n-val", type=int, default=60, help="held-out frames, excluded from training"
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.ann)
    train_stems, val_stems = split_stems(coco, args.n_val, args.split_seed)
    chosen = dedup_most_complete(coco)
    dataset = DetectorDataset(coco, train_stems, chosen, args.images, train_aug=True)
    print(
        f"{len(train_stems)} train frames ({len(dataset)} with instances), "
        f"{len(val_stems)} held out, device {device}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
    )

    model = build_detector(args.img_size, pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * f) for f in MILESTONE_FRACTIONS],
        gamma=0.1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start_epoch = 0
    resume = latest_checkpoint(args.out)
    if resume is not None:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print(f"resumed {resume}, next epoch {start_epoch}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, batches = 0.0, 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            batches += 1
        scheduler.step()
        print(
            f"epoch {epoch} loss {total / max(batches, 1):.4f} "
            f"elapsed {time.time() - started:.0f}s",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            args.out / f"epoch_{epoch:02d}.pt",
        )
        for old in sorted(args.out.glob("epoch_*.pt"))[: -args.keep_checkpoints]:
            old.unlink()

    print(
        f"done; detector = {args.out / f'epoch_{args.epochs - 1:02d}.pt'}", flush=True
    )


if __name__ == "__main__":
    main()
