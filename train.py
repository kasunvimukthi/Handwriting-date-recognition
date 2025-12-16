# train.py (UPGRADED)
import os
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import preprocess_32x32, augment_image
from model import create_char_cnn

LABELS = "char_dataset/labels.txt"
IMAGES  = "char_dataset/images"

# ------------------------------------------------------
# Load charset
# ------------------------------------------------------
def load_charset():
    chars = set()
    for line in open(LABELS, "r", encoding="utf-8"):
        if "\t" not in line:
            continue
        _, ch = line.strip().split("\t")
        chars.add(ch)
    return sorted(list(chars))

charset = load_charset()
print("CHARSET:", charset)

if len(charset) < 2:
    print("❌ Not enough samples to train.")
    exit()

# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
images = []
labels = []

for line in open(LABELS, "r", encoding="utf-8"):
    fname, ch = line.strip().split("\t")
    path = os.path.join(IMAGES, fname)

    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print("⚠ Missing image:", path)
        continue

    img = preprocess_32x32(raw)

    # Original sample
    images.append(img.astype("float32") / 255.0)
    labels.append(charset.index(ch))

    # --------------------------------------------------
    # Data Augmentation (x3 more samples per image)
    # --------------------------------------------------
    for _ in range(3):
        aug = augment_image(img)
        images.append(aug.astype("float32") / 255.0)
        labels.append(charset.index(ch))


# Convert to arrays
images = np.array(images).reshape(-1, 32, 32, 1)
labels = np.array(labels)

print("Total training samples after augmentation:", len(images))

# ------------------------------------------------------
# Create model
# ------------------------------------------------------
model = create_char_cnn(len(charset))

# ------------------------------------------------------
# Callbacks
# ------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ModelCheckpoint("char_cnn_best.h5", save_best_only=True, monitor="val_loss")
]

# ------------------------------------------------------
# TRAINING
# ------------------------------------------------------
# If dataset is too small for validation_split, disable it
if len(images) < 40:
    print("⚠ Dataset too small → disabling validation split")
    val_split = 0.0
else:
    val_split = 0.1

model.fit(
        images, labels,
        epochs=300,
        batch_size=16,
        validation_split=val_split,
        callbacks=callbacks
)

# Save final model
model.save("char_cnn.h5")
print("✔ Training complete")
print("✔ Saved: char_cnn.h5")
print("✔ Best: char_cnn_best.h5")
