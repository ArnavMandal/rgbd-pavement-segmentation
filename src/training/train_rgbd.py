#!/usr/bin/env python3
import os, warnings, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import PavementDataset
from models.unet import UNet

# ——— CONFIG ——————————————————————————————————————————————————————
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE          = 8
EPOCHS              = 30
LR                  = 1e-3
EARLY_STOP_PATIENCE = 5

TRAIN_CSV     = "data/splits/train.csv"
VAL_CSV       = "data/splits/val.csv"
RGB_DIR       = "data/processed/rgb"
DEPTH_DIR     = "data/processed/depth"    # ← your depth maps
MASK_DIR      = "data/processed/mask"

CHECKPOINT    = "models/best_unet_rgbd.pth"
LOG_SUBDIR    = "rgbd_fusion"
# ——————————————————————————————————————————————————————————————————————

def main():
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    writer = SummaryWriter(f"runs/{LOG_SUBDIR}")

    if DEVICE=="cpu":
        warnings.warn("CUDA is not available. Training on CPU may be slow.")
        num_workers = 2
    else:
        num_workers = 4

    print(">>> RGB-D Training configuration:")
    print(f"  Device:        {DEVICE}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Workers:       {num_workers}")

    # — create datasets & loaders —
    train_ds = PavementDataset(
        split_csv=TRAIN_CSV,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        depth_dir=DEPTH_DIR,
        split_prefix="train_"
    )
    val_ds = PavementDataset(
        split_csv=VAL_CSV,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        depth_dir=DEPTH_DIR,
        split_prefix="val_"
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(DEVICE!="cpu")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE!="cpu")
    )

    # — model / loss / optimizer / scheduler —
    model     = UNet(n_channels=4, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    best_val = float("inf")
    stall    = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            msks = batch["mask"].to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, msks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        writer.add_scalar("train/loss", avg_train, epoch)

        # — validation —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["image"].to(DEVICE))
                val_loss += criterion(logits, batch["mask"].to(DEVICE)).item()
        avg_val = val_loss / len(val_loader)
        writer.add_scalar("val/loss", avg_val, epoch)

        # — checkpoint & early stopping —
        if avg_val < best_val:
            best_val = avg_val
            stall = 0
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"Epoch {epoch}: new best val loss {avg_val:.4f}, saved → {CHECKPOINT}")
        else:
            stall += 1

        scheduler.step(avg_val)
        if stall >= EARLY_STOP_PATIENCE:
            print(f"No improvement for {stall} epochs → early stopping.")
            break

        print(f"Epoch {epoch} done. Val loss: {avg_val:.4f}")

    writer.close()

if __name__=="__main__":
    main()
