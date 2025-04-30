
import os, shutil

SRC_DIR = "data/raw"
DST_DIR = "data/raw_clean"

os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if not fname.lower().endswith(".png"):
        continue
    # expect names like "um_000123.png", "uu_000456.png", "umm_000789.png"
    parts = fname.split("_", 1)
    if len(parts) != 2:
        print(f"⚠️  skipping unexpected file: {fname}")
        continue
    new_name = parts[1]       # e.g. "000123.png"
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, new_name)
    print(f"Moving {fname} → {new_name}")
    shutil.move(src_path, dst_path)

print(f"\n✅ All renamed files are now in {DST_DIR}")
