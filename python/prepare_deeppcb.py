"""
LineTriage - DeepPCB to YOLO Dataset Converter

Converts the DeepPCB raw dataset (PCBData/ folder) into YOLOv8 format.
Reads: data/raw/PCBData/groupXXXXX/XXXXX/xxxxx_test.jpg
       data/raw/PCBData/groupXXXXX/XXXXX_not/xxxxx.txt
Writes: data/deeppcb_yolo/images/{train,val}/*.jpg
        data/deeppcb_yolo/labels/{train,val}/*.txt
        data/deeppcb_yolo/dataset.yaml
"""

import os
import shutil
import random
import cv2
import yaml
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
RAW_DATA_DIR = "data/raw/PCBData"
OUTPUT_DIR = "data/deeppcb_yolo"
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# DeepPCB class IDs: 1-6 → YOLO 0-5
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

# ============================================================
# SETUP
# ============================================================
random.seed(RANDOM_SEED)

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


# ============================================================
# CONVERTER
# ============================================================
def convert_annotation(txt_path, img_width, img_height):
    """
    Convert DeepPCB format (x1 y1 x2 y2 class_id) to YOLO format
    (class_id cx cy w h) — all normalized 0-1.
    """
    yolo_lines = []
    if not os.path.exists(txt_path):
        return yolo_lines

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                x1, y1, x2, y2, class_id = map(int, parts)
            except ValueError:
                continue

            if class_id == 0:  # Skip background
                continue

            yolo_class = class_id - 1

            cx = ((x1 + x2) / 2.0) / img_width
            cy = ((y1 + y2) / 2.0) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height

            # Clamp
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            yolo_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def main():
    print("=" * 60)
    print("DeepPCB → YOLO Dataset Conversion")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"\n⚠ ERROR: {RAW_DATA_DIR} not found.")
        print("   Please clone https://github.com/tangsanli5201/DeepPCB")
        print("   and copy its PCBData/ folder into data/raw/")
        return

    print(f"\nScanning {RAW_DATA_DIR}...")
    pairs = []

    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for fname in files:
            if fname.endswith("_test.jpg"):
                img_path = os.path.join(root, fname)
                base = fname.replace("_test.jpg", "")

                parent_dir = os.path.dirname(root)
                not_folder = os.path.basename(root) + "_not"
                ann_path = os.path.join(parent_dir, not_folder, base + ".txt")

                if os.path.exists(ann_path):
                    pairs.append((img_path, ann_path))

    print(f"Found {len(pairs)} image-annotation pairs")

    if len(pairs) == 0:
        print("\n⚠ No pairs found. Check folder structure.")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    class_counts = defaultdict(int)
    skipped = 0

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        print(f"\nProcessing {split_name}...")

        for img_path, ann_path in tqdm(split_pairs):
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            yolo_lines = convert_annotation(ann_path, w, h)

            if not yolo_lines:
                skipped += 1
                continue

            base = os.path.basename(img_path).replace("_test.jpg", "")

            shutil.copy(img_path,
                        os.path.join(OUTPUT_DIR, "images", split_name, f"{base}.jpg"))

            with open(os.path.join(OUTPUT_DIR, "labels", split_name, f"{base}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            for line in yolo_lines:
                cls = int(line.split()[0])
                class_counts[CLASS_NAMES[cls]] += 1

    # dataset.yaml
    yaml_config = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Train images: {len(os.listdir(os.path.join(OUTPUT_DIR, 'images', 'train')))}")
    print(f"Val images:   {len(os.listdir(os.path.join(OUTPUT_DIR, 'images', 'val')))}")
    print(f"Skipped:      {skipped}")
    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"\nDataset YAML: {OUTPUT_DIR}/dataset.yaml")


if __name__ == "__main__":
    main()