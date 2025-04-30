# src/training/train.py

import os
import warnings
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import PavementDataset
from models.unet import UNet

# ——— CONFIG ——————————————————————————————————————————————————————
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE           = 8
EPOCHS               = 30
LR                   = 1e-3
EARLY_STOP_PATIENCE  = 5   # stop if no val‐loss improvement for this many epochs

TRAIN_CSV            = "data/splits/train.csv"
VAL_CSV              = "data/splits/val.csv"
RGB_DIR              = "data/processed/rgb"
MASK_DIR             = "data/processed/mask"

CHECKPOINT_DIR       = "models"
LOG_DIR              = "runs/seg-experiment"
# ——————————————————————————————————————————————————————————————————————

def main():
    # make sure checkpoint dir exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # tensorboard writer
    writer = SummaryWriter(LOG_DIR)

    # CPU warning & dataloader workers
    if DEVICE == "cpu":
        warnings.warn("CUDA is not available. Training on CPU may be slow.")
        num_workers = 2
    else:
        num_workers = 4

    print("Training configuration:")
    print(f"- Device:           {DEVICE}")
    print(f"- Batch size:       {BATCH_SIZE}")
    print(f"- Learning rate:    {LR}")
    print(f"- Num workers:      {num_workers}")

    # — create datasets & loaders —
    train_ds = PavementDataset(TRAIN_CSV, RGB_DIR, MASK_DIR, split_prefix="train_")
    val_ds   = PavementDataset(VAL_CSV,   RGB_DIR, MASK_DIR, split_prefix="val_")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(DEVICE!="cpu")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE!="cpu")
    )

    # — model, loss, optimizer, scheduler —
    model     = UNet(n_channels=3, n_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True
    )

    best_val_loss  = float("inf")
    no_improve_cnt = 0
    global_step    = 0

    # — training loop —
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            msks = batch["mask"].to(DEVICE)

            logits = model(imgs)
            loss   = criterion(logits, msks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step  += 1

            if global_step % 20 == 0:
                avg_batch_loss = running_loss / 20
                writer.add_scalar("train/loss", avg_batch_loss, global_step)
                running_loss = 0.0

        # — validation pass —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                msks = batch["mask"].to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, msks).item()

        val_loss /= len(val_loader)
        writer.add_scalar("val/loss", val_loss, epoch)

        # — checkpoint & early stopping logic —
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            no_improve_cnt = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_unet.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {epoch}: new best val loss {val_loss:.4f}, checkpoint saved → {ckpt_path}")
        else:
            no_improve_cnt += 1

        # — step the LR scheduler —
        scheduler.step(val_loss)

        # — check for early stopping —
        if no_improve_cnt >= EARLY_STOP_PATIENCE:
            print(f"No improvement for {no_improve_cnt} epochs → early stopping.")
            break

        print(f"Epoch {epoch} done. Val loss: {val_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
