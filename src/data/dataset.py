# src/data/dataset.py

import os, csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PavementDataset(Dataset):
    def __init__(self, split_csv, rgb_dir, mask_dir, split_prefix="", img_size=(256,256)):
        """
        split_csv    : path to e.g. data/splits/train.csv
        rgb_dir      : e.g. data/processed/rgb
        mask_dir     : e.g. data/processed/mask
        split_prefix : if your files are named "train_000012.png", pass "train_"
                       if your files are just "000012.png", leave it as ""
        """
        self.rgb_dir     = rgb_dir
        self.mask_dir    = mask_dir
        self.split_pref  = split_prefix  or ""
        self.img_size    = img_size

        # read IDs from CSV
        with open(split_csv, newline="") as f:
            reader     = csv.DictReader(f)
            self.ids   = [row["id"] for row in reader]

        # image transforms: Rescale → ToTensor → Normalize
        self.tf_img = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # 0–255 → 0.0–1.0
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])

        # mask transforms: Rescale → ToTensor
        # (PIL L‐mode is 0–255; ToTensor divides by 255 → 0.0/1.0)
        self.tf_mask = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),  # 0–255 → 0.0–1.0
            transforms.Lambda(lambda x: (x > 0.5).float()),  
            # ensure strictly binary
        ])


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        fn = self.ids[idx]
        fname = f"{self.split_pref}{fn}.png"

        # load RGB
        img_path = os.path.join(self.rgb_dir, fname)
        img = Image.open(img_path).convert("RGB")

        # load grayscale mask
        msk_path = os.path.join(self.mask_dir, fname)
        msk = Image.open(msk_path).convert("L")

        # apply transforms
        img_t = self.tf_img(img)
        msk_t = self.tf_mask(msk)

        return {
            "image": img_t,     # FloatTensor[3,H,W], normalized
            "mask":  msk_t,     # FloatTensor[1,H,W], exactly 0 or 1
            "id":    fn
        }
