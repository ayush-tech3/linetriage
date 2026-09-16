"""
LineTriage - Main Python Bridge Server

Captures images from ESP32-CAM, runs YOLOv8 inference,
manages priority queue, controls conveyor + rejection servo.
"""

import requests
import cv2
import numpy as np
from ultralytics import YOLO
import heapq
import time
import yaml
from datetime import datetime
import os

# ============================================================
# CONFIG
# ============================================================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

ESP32_IP = config["esp32"]["ip"]
MODEL_PATH = config["model"]["path"]
CLASS_NAMES = config["model"]["class_names"]
SEVERITY_WEIGHTS = config["severity_weights"]
MAX_QUEUE_SIZE = config["queue"]["max_size"]
QUEUE_ALERT_THRESHOLD = config["queue"]["alert_threshold"]
NORMAL_SPEED = config["conveyor"]["normal_speed"]
SLOW_SPEED = config["conveyor"]["slow_speed"]
LOOP_DELAY = config["conveyor"]["loop_delay_seconds"]

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")


# ============================================================
# HARDWARE
# ============================================================
def capture_image_from_esp32():
    url = f"http://{ESP32_IP}/capture"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[HARDWARE] Capture failed: {e}")
    return None


def control_motor(cmd, value=None):
    url = f"http://{ESP32_IP}/motor"
    params = {"cmd": cmd}
    if value is not None:
        params["value"] = value
    try:
        return requests.get(url, params=params, timeout=3).text
    except Exception as e:
        print(f"[HARDWARE] Motor failed: {e}")
        return None


def trigger_rejection():
    url = f"http://{ESP32_IP}/reject"
    try:
        return requests.get(url, timeout=3).text
    except Exception as e:
        print(f"[HARDWARE] Rejection failed: {e}")
        return None


# ============================================================
# INFERENCE
# ============================================================
def predict_defects(image):
    results = model(image, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "class": CLASS_NAMES[cls_id],
                "confidence": conf,
                "bbox": xyxy,
                "severity": SEVERITY_WEIGHTS.get(CLASS_NAMES[cls_id], 1),
            })
    return detections


def apply_glare_correction(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# PRIORITY QUEUE
# ============================================================
class InspectionQueue:
    def __init__(self, max_size=MAX_QUEUE_SIZE):
        self.queue = []
        self.max_size = max_size
        self.counter = 0
        self.overflow_count = 0

    def add(self, det):
        priority = det["severity"] * det["confidence"]
        if len(self.queue) >= self.max_size:
            min_priority = min(item[0] for item in self.queue)
            if priority > -min_priority:
                self.queue = [item for item in self.queue if item[0] != min_priority]
                heapq.heapify(self.queue)
                self.overflow_count += 1
            else:
                return False, priority
        self.counter += 1
        heapq.heappush(self.queue, (-priority, self.counter, det))
        return True, priority

    def status(self):
        return {
            "size": len(self.queue),
            "max": self.max_size,
            "near_capacity": len(self.queue) >= QUEUE_ALERT_THRESHOLD,
            "overflows": self.overflow_count,
        }


inspection_queue = InspectionQueue()


def adaptive_threshold(qsize):
    t = config["queue"]["adaptive_thresholds"]
    if qsize >= MAX_QUEUE_SIZE:
        return t["saturated"]
    elif qsize >= QUEUE_ALERT_THRESHOLD:
        return t["warning"]
    return t["normal"]


# ============================================================
# PIPELINE
# ============================================================
def process_product(image, product_id):
    print(f"\n{'='*50}\nProduct #{product_id}\n{'='*50}")

    corrected = apply_glare_correction(image)
    detections = predict_defects(corrected)
    queue_status = inspection_queue.status()
    print(f"Queue: {queue_status['size']}/{queue_status['max']}")

    if not detections:
        print("✓ No defects detected. SAFE.")
        return {"product_id": product_id, "defect": None, "queued": False}

    top_defect = max(detections, key=lambda d: d["severity"] * d["confidence"])
    print(f"Top defect: {top_defect['class']} | "
          f"conf={top_defect['confidence']:.2%} | "
          f"severity={top_defect['severity']}/5")

    threshold = adaptive_threshold(queue_status["size"])
    is_critical = top_defect["severity"] >= 4

    if top_defect["confidence"] >= threshold and is_critical:
        print("→ CRITICAL: Triggering rejection servo")
        trigger_rejection()
    else:
        added, priority = inspection_queue.add(top_defect)
        if added:
            print(f"→ Queued for inspection (priority={priority:.2f})")
        else:
            print("→ Tolerated (queue full, low priority)")

    return {
        "product_id": product_id,
        "defect": top_defect["class"],
        "confidence": top_defect["confidence"],
        "queued": True,
    }


def run_continuous_pipeline():
    print("\n" + "=" * 60)
    print("LineTriage - Continuous Processing Pipeline")
    print("=" * 60)

    control_motor("start", NORMAL_SPEED)
    product_id = 0

    try:
        while True:
            product_id += 1
            print(f"\n[Product #{product_id}] Capturing...")
            image = capture_image_from_esp32()
            if image is None:
                time.sleep(2)
                continue

            process_product(image, product_id)

            if inspection_queue.status()["near_capacity"]:
                print("⚠ Queue near capacity - slowing conveyor")
                control_motor("speed", SLOW_SPEED)
            else:
                control_motor("speed", NORMAL_SPEED)

            time.sleep(LOOP_DELAY)
    except KeyboardInterrupt:
        print("\nStopping...")
        control_motor("stop")


if __name__ == "__main__":
    print(f"Testing ESP32 at {ESP32_IP}...")
    try:
        r = requests.get(f"http://{ESP32_IP}/status", timeout=3)
        print(f"✓ ESP32: {r.text}")
    except Exception as e:
        print(f"✗ ESP32 unreachable: {e}")

    run_continuous_pipeline()