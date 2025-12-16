# ✍️ Handwritten Date Recognition (CNN-based OCR)

A lightweight **handwritten date recognition system** built using **Python, OpenCV, and Keras (TensorFlow)**.  
This project focuses on recognizing **simple handwritten digits (0–9)** from real images and is optimized for **date formats** such as:


The system **learns incrementally from real user corrections** and improves accuracy over time.

---

## 📌 Key Features

- ✅ Handwritten digit recognition (0–9)
- ✅ Uses **real grayscale bitmap images only**
- ✅ Character segmentation using OpenCV
- ✅ CNN-based classification (32×32 grayscale)
- ✅ Self-learning pipeline (auto-add corrected samples)
- ✅ Automatic retraining after new data
- ❌ No synthetic fonts
- ❌ No skeletons / centerlines
- ❌ No fuzzy matching

---

## 🧠 Pipeline Overview

1. Input handwritten image
2. Segment characters
3. Preprocess each character
   - Grayscale
   - Crop to content
   - Square padding
   - Resize to 32×32
4. CNN prediction
5. Low-confidence predictions require user correction
6. Corrected samples are saved
7. Model retrains automatically
8. Accuracy improves progressively

---

## 📂 Project Structure
.
├── char_dataset/
│   ├── images/          # Saved 32×32 grayscale characters
│   └── labels.txt       # filename<TAB>label
│
├── infer.py             # Main OCR inference & self-learning
├── train.py             # CNN training with augmentation
├── model.py             # CNN architecture
├── preprocess.py        # Image preprocessing & augmentation
├── segment.py           # Character segmentation
├── fix_labels.py        # Sort labels file safely
├── README.md


---

## ⚙️ Requirements

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy

## Install dependencies
pip install opencv-python tensorflow numpy

## 🔹 Run OCR on an image
python infer.py test.png

* High-confidence predictions are accepted automatically
* Low-confidence predictions prompt user correction
* Corrections are stored and used for retraining

## 🔹 Train / Retrain the Model Manually
python train.py
* Normally not required, since retraining is triggered automatically after corrections.

## 🧪 Data Augmentation

Training includes light augmentation to improve generalization:

* Small rotations
* Minor shifts
* Noise injection

This helps the model learn variations in handwriting.

## 🧠 CNN Architecture (Summary)

* Input: 32×32×1
* Conv layers: 32 → 64 → 128
* MaxPooling
* Dense layer (256 units)
* Softmax output

Designed to be:
* Fast
* Lightweight
* Easy to extend

## 📈 Learning Strategy

* Starts with a very small dataset
* Learns only from real handwriting
* Optimized for numeric date formats
* Accuracy improves significantly after ~30–50 samples per digit

## 🚧 Limitations

* Not a full OCR engine
* Needs initial manual corrections
* Best suited for clean handwritten dates

## 📜 License
Open-source. Free to use for learning and experimentation.





