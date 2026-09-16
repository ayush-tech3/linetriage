"""
LineTriage - Single Image Test
Usage: python model_test.py <image_path>
"""

import sys
import os
import cv2
from ultralytics import YOLO
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
CLASS_NAMES = config["model"]["class_names"]
SEVERITY_WEIGHTS = config["severity_weights"]


def predict(image_path, model):
    results = model(image_path, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append((CLASS_NAMES[cls_id], conf, xyxy))
    return detections


def compute_priority(detections):
    if not detections:
        return 0.0
    return max(SEVERITY_WEIGHTS.get(c, 1) * conf for c, conf, _ in detections)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Not found: {image_path}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"Running inference on {image_path}...")
    detections = predict(image_path, model)

    if not detections:
        print("\n✓ No defects detected. Product is SAFE.")
    else:
        print(f"\n⚠ {len(detections)} defect(s) detected:")
        for cls, conf, bbox in detections:
            sev = SEVERITY_WEIGHTS.get(cls, 1)
            print(f"  - {cls} | conf={conf:.2%} | severity={sev}/5 | bbox={bbox}")

        priority = compute_priority(detections)
        print(f"\nHighest priority score: {priority:.2f}")
        if priority >= 3.5:
            print("→ CRITICAL: Reject via servo")
        elif priority >= 2.0:
            print("→ WARNING: Enqueue for inspection")
        else:
            print("→ Product continues")

    results = model(image_path, verbose=False)
    annotated = results[0].plot()
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/annotated_" + os.path.basename(image_path)
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")