"""The two training loops, run for real on synthetic data: one optimiser step of the detector and one
epoch of the refiner, whose checkpoint must then load through the inference loader. Needs no
competition data and downloads nothing (both networks are built without pretrained weights).

  python -m unittest discover tests
"""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from filseg.data import CropDataset
from filseg.model import CropUNetSeedbox, build_detector, load_refiner
from scripts.train_refiner import soft_dice_loss, train, validation_dice

CROP = 64


def crop_cache(directory: Path, n: int = 6):
    """A tiny stand-in for build_crop_cache's output: n three-channel crops with a bright bar whose
    mask the refiner can actually fit."""
    rng = np.random.default_rng(0)
    images = np.zeros((n, 3, CROP, CROP), dtype=np.uint8)
    masks = np.zeros((n, CROP, CROP), dtype=np.uint8)
    for i in range(n):
        y = 20 + (i % 3) * 5
        images[i, 0] = rng.integers(30, 90, (CROP, CROP), dtype=np.uint8)
        images[i, 0, y : y + 8, 10:50] = 220
        images[i, 1] = images[i, 0]
        images[i, 2, y - 4 : y + 12, 6:54] = 255
        masks[i, y : y + 8, 10:50] = 1
    image_path, mask_path = directory / "crops.npy", directory / "masks.npy"
    np.save(image_path, images)
    np.save(mask_path, masks)
    return image_path, mask_path


class TestLoss(unittest.TestCase):
    def test_soft_dice_is_zero_on_a_perfect_prediction(self):
        target = torch.zeros(1, 1, 8, 8)
        target[..., 2:6, 2:6] = 1
        logits = (target * 2 - 1) * 50  # +-50 -> sigmoid saturates
        self.assertLess(float(soft_dice_loss(logits, target)), 1e-3)
        self.assertGreater(float(soft_dice_loss(-logits, target)), 0.9)


class TestRefinerTraining(unittest.TestCase):
    def test_one_epoch_writes_a_checkpoint_the_inference_loader_accepts(self):
        torch.manual_seed(0)
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image_path, mask_path = crop_cache(tmp)
            loader = DataLoader(
                CropDataset(image_path, mask_path, augment=True), batch_size=3
            )
            val_loader = DataLoader(CropDataset(image_path, mask_path), batch_size=6)
            checkpoint = tmp / "best_refiner_seedbox.pt"
            history = train(
                CropUNetSeedbox(pretrained=False),
                loader,
                val_loader,
                epochs=1,
                lr=3e-4,
                weight_decay=1e-4,
                device=torch.device("cpu"),
                checkpoint=checkpoint,
            )
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["loss"]))
            self.assertTrue(checkpoint.exists())
            # the key test: the trainer's checkpoint layout is the one inference reads
            refiner = load_refiner(checkpoint, torch.device("cpu"))
            out = refiner(torch.zeros(1, 3, CROP, CROP))
            self.assertEqual(out.shape, (1, 1, CROP, CROP))

    def test_validation_dice_is_a_fraction(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            image_path, mask_path = crop_cache(Path(tmp))
            loader = DataLoader(CropDataset(image_path, mask_path), batch_size=6)
            dice = validation_dice(
                CropUNetSeedbox(pretrained=False), loader, torch.device("cpu")
            )
            self.assertGreaterEqual(dice, 0.0)
            self.assertLessEqual(dice, 1.0)


class TestDetectorTraining(unittest.TestCase):
    def test_one_optimiser_step_moves_the_weights(self):
        torch.manual_seed(0)
        model = build_detector(img_size=128, pretrained=False)
        model.train()
        image = torch.rand(3, 128, 128)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 30.0]]),
            "labels": torch.ones(1, dtype=torch.int64),
            "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
        }
        target["masks"][0, 10:30, 10:40] = 1
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4
        )
        before = model.roi_heads.mask_predictor.mask_fcn_logits.weight.detach().clone()

        losses = model([image], [target])
        total = sum(losses.values())
        self.assertIn("loss_mask", losses)
        self.assertTrue(torch.isfinite(total))
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        after = model.roi_heads.mask_predictor.mask_fcn_logits.weight
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
