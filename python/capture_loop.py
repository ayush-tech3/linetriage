import time
from pathlib import Path

import cv2
import numpy as np
import requests

# Update this address if the hotspot gives the ESP32 a different IP.
CAMERA_URL = "http://10.56.90.127/capture"

SAVE_FOLDER = Path("captures")
SAVE_FOLDER.mkdir(exist_ok=True)

print(f"Getting images from: {CAMERA_URL}")
print("Press Q in the image window to stop.")

try:
    while True:
        try:
            response = requests.get(CAMERA_URL, timeout=10)
            response.raise_for_status()

            image_bytes = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                print("The camera response was not a readable image.")
                time.sleep(1)
                continue

            filename = time.strftime("capture_%Y%m%d_%H%M%S.jpg")
            save_path = SAVE_FOLDER / filename
            cv2.imwrite(str(save_path), image)

            print(f"Saved: {save_path}")
            cv2.imshow("LineTriage - ESP32-CAM", image)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

            time.sleep(1)

        except requests.RequestException as error:
            print(f"Could not reach the camera: {error}")
            print("Check that the laptop and ESP32-CAM are on the same hotspot.")
            time.sleep(2)

finally:
    cv2.destroyAllWindows()
