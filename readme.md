# LineTriage — Cooperative PCB Defect Detection & Sorting

**Hackathon:** IEEE SYNAPSE 1.0
**Problem Statement:** PS-01 · SYN26-CS01 (Smart Manufacturing / Computer Vision)
**Dataset:** DeepPCB (tangsanli5201/DeepPCB)
**Model:** YOLOv8n (fine-tuned on DeepPCB)

---

## Overview

LineTriage is a complete edge-AI pipeline that detects PCB surface defects on a moving conveyor belt, ranks them by business severity, and physically rejects or queues them for human inspection — without ever overflowing the inspection queue or stopping the line.

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ESP32-CAM      │───▶│  Python Server   │───▶│  Priority Queue │───▶│  Actuation      │
│  (Image Capture)│    │  (YOLOv8 Inference)│  │  (Severity × Conf)│  │  (Servo + Motor)│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

### Dataset — DeepPCB

- **Source:** https://github.com/tangsanli5201/DeepPCB
- **Classes (6):** open, short, mousebite, spur, copper, pin-hole
- **Format:** Image pairs (template + tested) with `x1 y1 x2 y2 class_id` annotations
- **Size:** ~1,500 image pairs

### Benchmark Performance

| Metric | Value |
|--------|-------|
| mAP@50 | ~0.97 |
| mAP@50-95 | ~0.71 |
| F-score | ~0.97 |

---

## Setup Instructions

### 1. Clone / Prepare the Dataset

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git
mkdir -p data/raw
cp -r DeepPCB/PCBData data/raw/
cp -r DeepPCB/evaluation .
```

Your `data/raw/PCBData/` should look like:
```
PCBData/
├── groupXXXXX/
│   ├── XXXXX/            (contains _test.jpg and _temp.jpg files)
│   └── XXXXX_not/        (contains .txt annotation files)
├── trainval.txt
└── test.txt
```

### 2. Install Python Dependencies

```bash
pip install -r python/requirements.txt
```

### 3. Convert DeepPCB → YOLO Format

```bash
python python/prepare_deeppcb.py
```

Outputs to `data/deeppcb_yolo/`:
- `images/train/`, `images/val/`
- `labels/train/`, `labels/val/`
- `dataset.yaml`

### 4. Train the Model

```bash
python python/train_model.py
```

Best weights saved to `models/best.pt`.

### 5. Flash the ESP32-CAM

1. Open `arduino/linetriage_esp32cam.ino` in Arduino IDE
2. Board: **AI Thinker ESP32-CAM**
3. Install library: `ESP32Servo`
4. Update WiFi SSID/password
5. Upload (connect GPIO0 to GND during upload)
6. Note the IP address in Serial Monitor
7. Update `config/config.yaml` with that IP

### 6. Run the System

```bash
python python/line_triage_server.py
```

### 7. (Optional) Evaluate on Official DeepPCB Test Set

```bash
python python/evaluate_deeppcb.py
```

---

## Hardware Wiring

| Component | ESP32-CAM GPIO | Notes |
|-----------|----------------|-------|
| L298N IN1 | GPIO 12 | Motor direction |
| L298N IN2 | GPIO 13 | Motor direction |
| L298N ENA | GPIO 14 | PWM speed |
| Servo Signal | GPIO 15 | Rejection gate |
| Power | 5V 2A (external) | Do NOT power from USB |

---

## Priority Queue Policy

**Priority score** = `severity_weight × confidence`

**Severity weights:**
| Defect | Weight |
|--------|--------|
| open | 5 |
| short | 5 |
| mousebite | 4 |
| pin-hole | 4 |
| spur | 3 |
| copper | 3 |

**Adaptive threshold:**
| Queue state | Confidence threshold |
|-------------|----------------------|
| Queue < 4 | 0.50 |
| Queue = 4 | 0.70 |
| Queue ≥ 5 | 0.85 |

**Conveyor slows** to PWM 100 when queue ≥ 4 (never stops).

---

## License

MIT