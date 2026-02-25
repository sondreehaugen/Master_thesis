"""
Splitter images og labels i train/val-sett (80/20) og strukturerer
dem slik YOLO forventer:
  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/
"""

import os
import shutil
import random
from pathlib import Path

# ── Konfigurasjon ─────────────────────────────────────────────────────────────
SEED = 42
TRAIN_RATIO = 0.8

BASE_DIR    = Path(__file__).parent
SRC_IMAGES  = BASE_DIR / "images"
SRC_LABELS  = BASE_DIR / "labels"
DATASET_DIR = BASE_DIR / "dataset"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    random.seed(SEED)

    # find all images and match with labels
    image_files = sorted(SRC_IMAGES.glob("*.jpg"))
    paired = []
    missing_labels = []

    for img_path in image_files:
        label_path = SRC_LABELS / (img_path.stem + ".txt")
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"Missing label for {len(missing_labels)} pictures, skipped:")
        for f in missing_labels:
            print(f"   {f}")

    print(f"\n Found {len(paired)} pictures with labels.")

    # Shuffle og split dataset
    random.shuffle(paired)
    split_idx = int(len(paired) * TRAIN_RATIO)
    train_pairs = paired[:split_idx]
    val_pairs   = paired[split_idx:]

    print(f"   Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # make directories
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # copy files
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, DATASET_DIR / "images" / split / img_path.name)
            shutil.copy2(lbl_path, DATASET_DIR / "labels" / split / lbl_path.name)

    print(f"\n dataset ready in:  {DATASET_DIR}")
    print("   dataset/images/train/")
    print("   dataset/images/val/")
    print("   dataset/labels/train/")
    print("   dataset/labels/val/")


if __name__ == "__main__":
    main()
