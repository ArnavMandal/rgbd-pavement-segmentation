import os
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

def binarize_mask(x):
    """Convert mask to binary by thresholding at 0.5"""
    return (x > 0.5).float()

class PavementDataset(Dataset):
    def __init__(
        self,
        split_csv:    str,
        rgb_dir:      str,
        mask_dir:     str,
        depth_dir:    str = None,
        split_prefix: str = "",
        img_size:     tuple = (256,256),
    ):
        """
        split_csv    : path to CSV with columns [split,id]
        rgb_dir      : folder of 8-bit RGB PNGs
        mask_dir     : folder of 8-bit binary mask PNGs
        depth_dir    : folder of 8-bit depth map PNGs
        split_prefix : e.g. "train_", "val_", or "" if none
        img_size     : (width, height) for resize
        """
        self.rgb_dir      = rgb_dir
        self.mask_dir     = mask_dir
        self.depth_dir    = depth_dir
        self.split_prefix = split_prefix or ""
        self.img_size     = img_size
        self.use_depth    = depth_dir is not None

        # read the list of IDs
        with open(split_csv, newline="") as f:
            reader   = csv.DictReader(f)
            self.ids = [row["id"] for row in reader]

        # transforms for RGB normalization (3-channel)
        self.tf_rgb = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),  # 0–255 → 0.0–1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        
        # transforms for RGBD normalization (4-channel)
        if self.use_depth:
            # For RGBD, we'll create tensors manually and only need normalization
            self.rgbd_normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5],  # depth mean=0.5
                std=[0.229, 0.224, 0.225, 0.5]    # depth std=0.5
            )

        # mask stays binary
        self.tf_mask = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),           # 0–255 → 0.0–1.0
            transforms.Lambda(binarize_mask),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fn    = self.ids[idx]
        fname = f"{self.split_prefix}{fn}.png"

        # load RGB
        img = Image.open(os.path.join(self.rgb_dir, fname)).convert("RGB")

        if self.use_depth:
            # Load and process depth (for RGBD mode)
            dpt = Image.open(os.path.join(self.depth_dir, fname)).convert("L")
            dpt = dpt.resize(self.img_size, Image.NEAREST)
            
            # stack into 4-channel tensor
            rgb_t = transforms.ToTensor()(img)    # [3,H,W]
            depth_t = transforms.ToTensor()(dpt)  # [1,H,W]
            img_t = torch.cat([rgb_t, depth_t], dim=0)  # [4,H,W]
            # normalize all 4 channels
            img_t = self.rgbd_normalize(img_t)  # normalize 4 channels
        else:
            # RGB-only mode
            img_t = self.tf_rgb(img)  # [3,H,W] normalized

        # load & process mask
        msk = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        msk = msk.resize(self.img_size, Image.NEAREST)
        msk_t = self.tf_mask(msk)     # [1,H,W], 0 or 1

        return {
            "image": img_t,   # FloatTensor[3,H,W] or [4,H,W]
            "mask":  msk_t,   # FloatTensor[1,H,W]
            "id":    fn
        }
