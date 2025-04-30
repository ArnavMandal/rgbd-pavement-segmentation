import os, glob, cv2, csv
from sklearn.model_selection import train_test_split

#here are the clean folders with raw data to
#be preprocessed
RAW_IMG_DIR = "data/raw_clean"
RAW_MASK_DIR = "data/raw_masks_clean"

#where to keep the processed data
OUT_IMG_DIR      = "data/processed/rgb"
OUT_MASK_DIR     = "data/processed/mask"

#split defintions to write
OUT_SPLIT_DIR = "data/splits"

IMG_SIZE = (256, 256)


#Train/Val/Test proportions
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

#checking if the output directories exist
for d in (OUT_IMG_DIR, OUT_MASK_DIR, OUT_SPLIT_DIR):
    os.makedirs(d, exist_ok=True)

# Gather all IDs (basename without extension)
img_ids = {
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(RAW_IMG_DIR,  "*.png"))
}
mask_ids = {
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(RAW_MASK_DIR, "*.png"))
}
# make sure each img id has a corresponding mask 
# to make a set of valid ids
all_ids = sorted(img_ids & mask_ids)
print(f"✅ {len(img_ids)} RGBs, {len(mask_ids)} masks → using {len(all_ids)} paired IDs")

#create the splits
train_ids, temp_ids = train_test_split(all_ids, train_size=TRAIN_FRAC,  random_state=42)

# assigns the leftover ids not assigned to train 
# and making them the validation and test set ids
val_ids, test_ids = train_test_split(
    temp_ids,
    train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
    random_state=42
)

# helper function to write a split CSV
def write_split_csv(split_name, id_list):
    path = os.path.join(OUT_SPLIT_DIR, f"{split_name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split","id"])
        for i in id_list:
            writer.writerow([split_name, i])
    print(f"Wrote {len(id_list)} IDs to {path}")

#csv dump
write_split_csv("train", train_ids)
write_split_csv("val",   val_ids)
write_split_csv("test",  test_ids)

#function to process a split and save it

def process_split(id_list, split_name):
    for idx in id_list:
        #read raw
        img = cv2.imread(os.path.join(RAW_IMG_DIR,  f"{idx}.png"))
        msk = cv2.imread(os.path.join(RAW_MASK_DIR, f"{idx}.png"), cv2.IMREAD_GRAYSCALE)

        #resize
        img_r = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        msk_r = cv2.resize(msk, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        #write out split
        cv2.imwrite(os.path.join(OUT_IMG_DIR,  f"{split_name}_{idx}.png"), img_r)
        cv2.imwrite(os.path.join(OUT_MASK_DIR, f"{split_name}_{idx}.png"), msk_r)

    print(f"Finished writing {len(id_list)} items for split '{split_name}'")

# Process each split
process_split(train_ids, "train")
process_split(val_ids,   "val")
process_split(test_ids,  "test")


print("\n\nPreprocessing completed")