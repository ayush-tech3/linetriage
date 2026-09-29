"""
LineTriage - YOLOv8 Training on DeepPCB

Trains YOLOv8n for 6-class PCB defect detection.
Expected: mAP50 ≈ 0.97
"""

import os
import shutil
import yaml
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_YAML = os.path.join(config["training"]["data_dir"], "dataset.yaml")
EPOCHS = config["training"]["num_epochs"]
BATCH = config["training"]["batch_size"]
IMG_SIZE = config["training"]["img_size"]
PATIENCE = config["training"]["patience"]
MODEL_OUTPUT_DIR = "models"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("LineTriage - YOLOv8 Training on DeepPCB")
    print("=" * 60)
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Batch:   {BATCH}")
    print(f"ImgSize: {IMG_SIZE}")
    print("=" * 60)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        device='cpu',
        project="pcb_defect_detection",
        name="yolov8n_deeppcb",
        pretrained=True,
        optimizer="auto",
        cos_lr=False,
        close_mosaic=10,
        seed=0,
        deterministic=True,
    )

    best_pt = "pcb_defect_detection/yolov8n_deeppcb/weights/best.pt"
    if os.path.exists(best_pt):
        shutil.copy(best_pt, os.path.join(MODEL_OUTPUT_DIR, "best.pt"))
        print(f"\n✓ Best model copied to {MODEL_OUTPUT_DIR}/best.pt")

    print("\nEvaluating on validation set...")
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    try:
        onnx_path = model.export(format="onnx")
        print(f"✓ ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")


if __name__ == "__main__":
    main()