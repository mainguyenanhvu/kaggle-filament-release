"""Train the Mask2Former voter of the vote: COCO-pretrained Swin-S on 1024 x 1024 full-resolution crops.

  python scripts/train_mask2former.py --ann <annotations.json> --images <train_images> --out checkpoints/m2f

Recipe: one record per frame (the most complete), one class, AdamW with weight decay 0.05, lr 1e-4 and
1e-5 on the backbone, one-cycle cosine schedule over 40 epochs, batch 2, gradient norm clipped at 1.0.
A crop is centred on a random instance 70% of the time; instances left under 20 px in a crop are dropped.
Besides the 60 held-out frames, 40 pool frames stay out of training to choose the score threshold.
Writes `final.pt` ({"model": state_dict, "epoch": int}) and a resumable `last.pt`.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.data import DetectorDataset, dedup_most_complete, load_coco, split_stems  # noqa: E402
from filseg.mask2former import build_pretrained, normalise  # noqa: E402

CROP = 1024
N_POOL_VAL = 40


def random_crop(image: torch.Tensor, masks: torch.Tensor):
    _, h, w = image.shape
    if masks.shape[0] and random.random() < 0.7:
        ys, xs = torch.nonzero(masks[random.randrange(masks.shape[0])], as_tuple=True)
        cy, cx = int(ys.float().mean()), int(xs.float().mean())
        y0 = int(np.clip(cy - CROP // 2 + random.randint(-CROP // 3, CROP // 3), 0, h - CROP))
        x0 = int(np.clip(cx - CROP // 2 + random.randint(-CROP // 3, CROP // 3), 0, w - CROP))
    else:
        y0, x0 = random.randint(0, h - CROP), random.randint(0, w - CROP)
    crop = masks[:, y0 : y0 + CROP, x0 : x0 + CROP]
    return image[:, y0 : y0 + CROP, x0 : x0 + CROP], crop[crop.flatten(1).sum(1) >= 20]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.ann)
    train_stems, _ = split_stems(coco)
    train_stems = train_stems[N_POOL_VAL:]
    dataset = DetectorDataset(coco, train_stems, dedup_most_complete(coco), args.images, train_aug=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda batch: batch,
        drop_last=True,
    )
    model = build_pretrained().to(device)
    backbone = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" in n]
    rest = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" not in n]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * 0.1}, {"params": rest, "lr": args.lr}], weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        total_steps=args.epochs * len(loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    start = 0
    last = args.out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["epoch"] + 1
    model.train()
    started = time.time()
    for epoch in range(start, args.epochs):
        losses = []
        for batch in loader:
            pixels, masks, classes = [], [], []
            for image, target in batch:
                crop, crop_masks = random_crop(image, target["masks"])
                pixels.append(normalise(crop))
                masks.append(crop_masks.float().to(device))
                classes.append(torch.zeros(crop_masks.shape[0], dtype=torch.long, device=device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(
                    pixel_values=torch.stack(pixels).to(device), mask_labels=masks, class_labels=classes
                ).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss))
        print(f"epoch {epoch} loss {np.mean(losses):.4f} elapsed {time.time() - started:.0f}s", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            last,
        )
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1}, args.out / "final.pt")
    print(f"done; voter = {args.out / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
