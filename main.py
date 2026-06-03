import cv2
import numpy as np
import os
from datetime import datetime

# CCTV Video File
VIDEO_PATH = "sample_cctv.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

os.makedirs("alerts", exist_ok=True)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Processing Completed")
        break

    mask = fgbg.apply(frame)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious = False

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 3000:

            suspicious = True

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "Suspicious Activity",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    if suspicious:

        cv2.putText(
            frame,
            "ALERT!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(
            f"alerts/alert_{timestamp}.jpg",
            frame
        )

    cv2.imshow(
        "AI Vision Shield - CCTV Detection",
        frame
    )

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
