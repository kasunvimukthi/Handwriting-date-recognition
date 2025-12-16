# segment.py
import cv2
import numpy as np

def segment_characters(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Strong binarization
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Helps detect slanted slashes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    dil = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w < 5 or h < 15:
            continue

        boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    for (x, y, w, h) in boxes:
        crop = gray[y:y+h, x:x+w]
        chars.append(crop)

    return chars
