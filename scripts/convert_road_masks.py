# scripts/convert_road_masks.py
import os
import glob
import cv2

RAW_MASK_DIR   = "data/raw_masks"        # um_road_*.png files
CLEAN_MASK_DIR = "data/raw_masks_clean"  #  binary masks 00001.png, 00002.png etc 

os.makedirs(CLEAN_MASK_DIR, exist_ok=True)

# grab all files that represent 'road' masks
pattern = os.path.join(RAW_MASK_DIR, "*_road_*.png")
for src_path in glob.glob(pattern):
    filename = os.path.basename(src_path)
    # e.g. filename = "um_road_000123.png" or "uu_road_000123.png"
    # we want the trailing "000123.png"
    frame_id = filename.split("_")[-1]

    # load, threshold to binary
    mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # any non-zero pixel becomes 255
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    dst_path = os.path.join(CLEAN_MASK_DIR, frame_id)
    cv2.imwrite(dst_path, binary)

print(f"Converted {len(glob.glob(pattern))} masks → {CLEAN_MASK_DIR}")
