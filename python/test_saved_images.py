from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

# Project ke main folder se script chalao.
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = YOLO(config["model"]["path"])
image_folder = Path("captures")

images = sorted(
    path for path in image_folder.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    print("captures folder mein koi image nahi mili.")
    raise SystemExit

for image_path in images:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Image open nahi hui: {image_path}")
        continue

    results = model(image, verbose=False)
    preview = results[0].plot()

    print(f"\nImage: {image_path.name}")
    if len(results[0].boxes) == 0:
        print("Koi defect detect nahi hua.")
    else:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"{class_name}: {confidence:.1%}")

    cv2.imshow("LineTriage model test - Q dabao agali photo ke liye", preview)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()