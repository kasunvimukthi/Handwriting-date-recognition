# preprocess.py
import cv2
import numpy as np
import random

def preprocess_32x32(img):
    """
    Takes a grayscale cropped character bitmap
    → pads to square
    → resizes to 32×32 grayscale.
    """
    g = img.astype(np.uint8)

    h, w = g.shape
    side = max(h, w)

    # create white square
    sq = np.full((side, side), 255, dtype=np.uint8)

    # center crop
    y = (side - h) // 2
    x = (side - w) // 2
    sq[y:y+h, x:x+w] = g

    # final resize
    out = cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)
    return out

def augment(img):
    """
    Apply light augmentations to improve training.
    Used only for training, not for inference.
    """

    h, w = img.shape

    # Random rotation (-10° to 10°)
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # Random elastic warp (tiny)
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # Add noise
    noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
    img = cv2.add(img, noise)

    return img

def augment_image(img):
    """
    Light augmentation for handwriting training.
    Input: 32x32 grayscale
    Output: 32x32 augmented grayscale
    """
    h, w = img.shape

    # Rotate
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # Translation
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    M2 = np.float32([[1, 0, dx], [0, 1, dy]])
    aug = cv2.warpAffine(aug, M2, (w, h), borderValue=255)

    # Noise
    noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
    aug = cv2.add(aug, noise)

    return aug