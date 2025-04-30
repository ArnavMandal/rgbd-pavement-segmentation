#!/usr/bin/env python3
import os
import re
import csv
import random
import shutil
from datetime import datetime

def extract_ids_from_filenames(directory):
    """Extract all IDs from PNG files in the directory."""
    ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            match = re.search(r'test_(\d+)\.png', filename)
            if match:
                ids.append(match.group(1))
    return sorted(ids)

def backup_original_files(directory):
    """Backup original CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(directory, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    
    for filename in ["train.csv", "val.csv", "test.csv"]:
        src_path = os.path.join(directory, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(backup_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Backed up {src_path} to {dst_path}")

def main():
    # Directory paths
    rgb_dir = "data/processed/rgb"
    splits_dir = "data/splits"
    
    # Extract IDs from files
    image_ids = extract_ids_from_filenames(rgb_dir)
    if not image_ids:
        print("No image files found in the directory!")
        return
    
    print(f"Found {len(image_ids)} image files")
    
    # Shuffle and split IDs
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)
    
    # 80% for training, 20% for validation
    split_idx = int(len(image_ids) * 0.8)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    # Backup original files
    backup_original_files(splits_dir)
    
    # Create new CSV files
    with open(os.path.join(splits_dir, "train.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "id"])
        for id in train_ids:
            writer.writerow(["train", id])
    
    with open(os.path.join(splits_dir, "val.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "id"])
        for id in val_ids:
            writer.writerow(["val", id])
    
    print(f"Created new train.csv with {len(train_ids)} samples")
    print(f"Created new val.csv with {len(val_ids)} samples")
    print("Training IDs:", train_ids)
    print("Validation IDs:", val_ids)

if __name__ == "__main__":
    main()

