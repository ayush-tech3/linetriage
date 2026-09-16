"""
LineTriage - DeepPCB Official Evaluation

Computes mAP and F-score using the official DeepPCB evaluation scripts.
Requires: evaluation/script.py, evaluation/rrc_evaluation_funcs.py, evaluation/gt.zip
"""

import os
import zipfile
import shutil
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
VAL_IMAGES_DIR = "data/deeppcb_yolo/images/val"
OUTPUT_ZIP = "logs/detection_results.zip"
GT_ZIP = "evaluation/gt.zip"
EVAL_SCRIPT = "evaluation/script.py"

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def generate_submission(model):
    """Run inference on val images, save in DeepPCB submission format."""
    print("Running inference on validation set...")
    results = model(VAL_IMAGES_DIR, verbose=False)

    results_dict = {}
    for r in results:
        base = os.path.splitext(os.path.basename(r.path))[0]
        lines = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls_name = CLASS_NAMES[cls_id]
            lines.append(f"{int(x1)},{int(y1)},{int(x2)},{int(y2)},{conf:.6f},{cls_name}")
        results_dict[base] = "\n".join(lines)

    os.makedirs("logs/tmp_submission", exist_ok=True)
    for base, content in results_dict.items():
        with open(f"logs/tmp_submission/{base}.txt", "w") as f:
            f.write(content)

    os.makedirs(os.path.dirname(OUTPUT_ZIP), exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir("logs/tmp_submission"):
            zf.write(os.path.join("logs/tmp_submission", fname), arcname=fname)

    shutil.rmtree("logs/tmp_submission")
    print(f"✓ Submission ZIP: {OUTPUT_ZIP}")
    return OUTPUT_ZIP


def run_official_eval(submission_zip):
    if not os.path.exists(GT_ZIP):
        print(f"\n⚠ Ground truth not found at {GT_ZIP}")
        return

    if not os.path.exists(EVAL_SCRIPT):
        print(f"\n⚠ Evaluation script not found at {EVAL_SCRIPT}")
        return

    cmd = f"python {EVAL_SCRIPT} -g={GT_ZIP} -s={submission_zip}"
    print(f"\nRunning: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    submission = generate_submission(model)
    run_official_eval(submission)