# src/evaluation/evaluate.py

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import PavementDataset
from models.unet import UNet

def compute_iou(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Compute average IoU over a batch."""
    B = preds.size(0)
    preds = preds.view(B, -1)
    masks = masks.view(B, -1)
    intersection = (preds * masks).sum(dim=1)
    union        = (preds + masks - preds * masks).sum(dim=1)
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def compute_dice(preds: torch.Tensor, masks: torch.Tensor) -> float:
    """Compute average Dice (F1) over a batch."""
    B = preds.size(0)
    preds = preds.view(B, -1)
    masks = masks.view(B, -1)
    intersection = (preds * masks).sum(dim=1)
    return ((2*intersection + 1e-6) / (preds.sum(dim=1) + masks.sum(dim=1) + 1e-6)).mean().item()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--split_csv",   required=True,
                   help="Path to test split CSV")
    p.add_argument("--rgb_dir",     default="data/processed/rgb")
    p.add_argument("--mask_dir",    default="data/processed/mask")
    p.add_argument("--depth_dir",   default="data/processed/depth",
                   help="Provide only if n_channels=4")
    p.add_argument("--n_channels",  type=int, choices=[3,4], default=4,
                   help="3 for RGB-only, 4 for RGB+D")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet(n_channels=args.n_channels, n_classes=1).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # build dataset
    ds = PavementDataset(
        split_csv   = args.split_csv,
        rgb_dir     = args.rgb_dir,
        mask_dir    = args.mask_dir,
        depth_dir   = args.depth_dir if args.n_channels == 4 else None,
        split_prefix= "test_",
        img_size    = (256,256),
    )
    dl = DataLoader(ds,
                    batch_size   = args.batch_size,
                    shuffle      = False,
                    num_workers  = args.num_workers,
                    pin_memory   = (device.type=="cuda"))

    ious, dices = [], []

    with torch.no_grad():
        for batch in dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(imgs)
            preds  = (torch.sigmoid(logits) > 0.5).float()

            ious.append(compute_iou(preds, masks))
            dices.append(compute_dice(preds, masks))

    print(f"\n==== Results ({args.n_channels}-channel) ====")
    print(f"Mean IoU : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}\n")

if __name__=="__main__":
    main()
