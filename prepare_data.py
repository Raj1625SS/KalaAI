import os
import shutil
import random
from pathlib import Path

# Configuration
CLASSES = ["mughal", "sanganeri", "kalamkari", "gamthi"]
RAW_DIR = "data/raw"
SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}
SEED = 42

random.seed(SEED)

for class_name in CLASSES:
    raw_folder = Path(RAW_DIR) / class_name
    
    if not raw_folder.exists():
        print(f"WARNING: {raw_folder} not found, skipping...")
        continue

    # Get all images
    images = [
        f for f in raw_folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".avif"]
    ]
    random.shuffle(images)
    total = len(images)

    # Calculate split sizes
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"\n{class_name} — total: {total}")
    for split, files in split_data.items():
        dest_folder = Path(f"data/{split}/{class_name}")
        dest_folder.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(files):
            ext = src.suffix.lower()
            dst = dest_folder / f"{class_name}_{i:04d}{ext}"
            shutil.copy2(src, dst)

        print(f"  {split}: {len(files)} images")

print("\nDone! Dataset is ready.")