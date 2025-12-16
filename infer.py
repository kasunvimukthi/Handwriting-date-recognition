# infer.py
import os, sys, uuid, cv2, numpy as np
from keras.models import load_model

from segment import segment_characters
from preprocess import preprocess_32x32

LABELS = "char_dataset/labels.txt"
IMAGES = "char_dataset/images"
MODEL  = "char_cnn_best.h5"

def init_dataset():
    os.makedirs(IMAGES, exist_ok=True)
    if not os.path.exists(LABELS):
        open(LABELS, "w").close()

def load_charset():
    chars = set()
    for line in open(LABELS, "r", encoding="utf-8"):
        if "\t" not in line: continue
        _, ch = line.strip().split("\t")
        chars.add(ch)
    return sorted(list(chars))

def save_sample(img32, label):
    fname = f"{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(IMAGES, fname), img32)

    with open(LABELS, "a", encoding="utf-8") as f:
        f.write(f"{fname}\t{label}\n")

def preview(img):
    cv2.imshow("preview", cv2.resize(img, (200,200), cv2.INTER_NEAREST))
    cv2.waitKey(1)

def main(img_path):
    init_dataset()

    charset = load_charset()

    # Load model or fallback to correction-only mode
    model = None
    if os.path.exists(MODEL) and charset:
        model = load_model(MODEL)
        print("✔ Model loaded")
    else:
        print("⚠ No model → correction mode only")

    crops = segment_characters(img_path)
    if not crops:
        print("No characters found.")
        return

    result = ""
    corrected = False

    for i, crop in enumerate(crops, 1):
        img32 = preprocess_32x32(crop)

        arr = img32.astype("float32") / 255.0
        arr = arr.reshape(1,32,32,1)

        pred, conf = "", 0.0

        if model:
            probs = model.predict(arr, verbose=0)[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            pred = charset[idx]
            print(f"Char {i}: {pred} ({conf*100:.1f}%)")

        # ask for correction if low confidence
        if conf < 0.95:
            preview(img32)
            user = input(f"Correct #{i} (Enter={pred}): ").strip()
            cv2.destroyAllWindows()

            if user == "":
                user = pred

            save_sample(img32, user)
            corrected = True
            result += user
        else:
            result += pred

    if corrected:
        print("New samples added → retraining")
        import subprocess
        subprocess.call([sys.executable, "fix_labels.py"])
        subprocess.call([sys.executable, "train.py"])

    print("\nFINAL:", result)

if __name__ == "__main__":
    main(sys.argv[1])
