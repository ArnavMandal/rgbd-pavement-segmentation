#!/usr/bin/env python3
# src/data_preprocessing/rename_depth_maps.py
"""
Rename depth maps according to the splits in the CSV files.
"""

import os
import csv
import shutil

# Paths
SPLITS_DIR = "data/splits"
DEPTH_DIR = "data/processed/depth"

def main():
    # Ensure depth directory exists
    if not os.path.exists(DEPTH_DIR):
        print(f"Error: Depth directory {DEPTH_DIR} does not exist!")
        return
    
    # Process each split
    splits = ["train", "val", "test"]
    for split in splits:
        csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
        
        # Skip if CSV file doesn't exist
        if not os.path.exists(csv_path):
            print(f"Warning: Split file {csv_path} not found. Skipping.")
            continue
        
        # Read IDs from CSV
        ids = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            ids = [row["id"] for row in reader]
        
        print(f"Processing {len(ids)} IDs for {split} split...")
        
        # Rename each depth map
        for id_str in ids:
            src_path = os.path.join(DEPTH_DIR, f"{id_str}.png")
            dst_path = os.path.join(DEPTH_DIR, f"{split}_{id_str}.png")
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied: {id_str}.png → {split}_{id_str}.png")
            else:
                print(f"Warning: Source file {src_path} not found.")
    
    print("\nDepth map renaming completed!")

if __name__ == "__main__":
    main()

