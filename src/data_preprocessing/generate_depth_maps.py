# src/data_preprocessing/generate_depth_maps.py
import os
import glob
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor

# 1) MODEL SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # or MiDaS for the larger model
midas.to(device).eval()
# MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# 2) PATHS
IN_DIR  = "data/raw_clean"
OUT_DIR = "data/processed/depth"
os.makedirs(OUT_DIR, exist_ok=True)

# 3) LOOP
for img_path in sorted(glob.glob(os.path.join(IN_DIR, "*.png"))):
    filename = os.path.basename(img_path)
    
    # read & preprocess
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms(img).to(device)
    
    # forward
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # normalize depth to 0–255 and save
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_uint8 = (depth * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, filename), depth_uint8)

    print(f"Saved depth: {filename}")
