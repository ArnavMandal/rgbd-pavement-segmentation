#!/usr/bin/env python3
import os
import csv
import re

def get_ids_from_csv(file_path):
    """Extract all IDs from a CSV file"""
    ids = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row['id'])
    return ids

def get_files_from_directory(dir_path):
    """Get all PNG files from directory and extract IDs"""
    file_ids = []
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return file_ids
        
    for filename in os.listdir(dir_path):
        if filename.endswith('.png'):
            # Extract ID from filename (e.g., test_000012.png -> 000012)
            match = re.search(r'test_(\d+)\.png', filename)
            if match:
                file_ids.append(match.group(1))
    return file_ids

def main():
    # Paths
    train_csv = "data/splits/train.csv"
    val_csv = "data/splits/val.csv"
    rgb_dir = "data/processed/rgb"
    
    # Get IDs from CSVs
    train_ids = get_ids_from_csv(train_csv)
    val_ids = get_ids_from_csv(val_csv)
    all_csv_ids = train_ids + val_ids
    
    print(f"Found {len(train_ids)} IDs in train.csv")
    print(f"Found {len(val_ids)} IDs in val.csv")
    print(f"Total of {len(all_csv_ids)} IDs from CSV files")
    
    # Get available files
    available_ids = get_files_from_directory(rgb_dir)
    print(f"Found {len(available_ids)} image files in {rgb_dir}")
    
    # Check for missing IDs
    missing_ids = [id for id in all_csv_ids if id not in available_ids]
    
    print(f"\nMissing files: {len(missing_ids)} out of {len(all_csv_ids)}")
    
    # Print first few missing IDs
    if missing_ids:
        print("First 10 missing IDs:")
        for id in missing_ids[:10]:
            print(f"  - {id}")
    
    # Check for extra files not in CSVs
    extra_ids = [id for id in available_ids if id not in all_csv_ids]
    print(f"\nExtra files not in CSVs: {len(extra_ids)}")
    
    if extra_ids:
        print("First 10 extra IDs:")
        for id in extra_ids[:10]:
            print(f"  - {id}")
            
    # Calculate valid IDs
    valid_ids = [id for id in all_csv_ids if id in available_ids]
    print(f"\nValid IDs (in both CSV and directory): {len(valid_ids)}")
    
    # Suggest solution
    if missing_ids:
        print("\nPossible solutions:")
        print("1. Update CSV files to only include IDs that have corresponding image files")
        print("2. Copy missing image files to the data directory")
        print("3. Update dataset class to handle missing files")

if __name__ == "__main__":
    main()

