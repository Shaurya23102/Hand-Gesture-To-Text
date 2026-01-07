

# Hand Gesture to Text Conversion (ASL)

## Overview

This project converts **hand gestures from the American Sign Language (ASL) dataset into text in real time**.
It uses **MediaPipe** for accurate hand detection and **MobileNet** for gesture classification. The detected hand region is cropped, passed to the trained model, and the predicted **ASL letter with its confidence score** is displayed on the screen.

---

## Features

* Real-time hand detection using MediaPipe
* Robust hand region extraction (ROI-based)
* Lightweight and efficient MobileNet-based classifier
* Displays predicted ASL letter with probability
* Works with live webcam feed

---

## Tech Stack

* **Python**
* **MediaPipe** – Hand landmark detection
* **TensorFlow**
* **MobileNet** – Gesture classification
* **OpenCV** – Video capture and visualization
* **NumPy**

---

## Workflow

1. Capture live video from webcam
2. Detect hand landmarks using MediaPipe
3. Extract and crop the hand region (ROI)
4. Resize and preprocess the cropped image
5. Pass the image to the MobileNet model
6. Predict ASL letter and confidence score
7. Display result above the detected hand
![image alt](https://github.com/Shaurya23102/Hand-Gesture-To-Text/blob/main/fig_1.png?raw=true)
---

## System Architecture

```
Webcam Input
     ↓
MediaPipe Hand Detection
     ↓
Hand Region Cropping (ROI)
     ↓
Image Preprocessing
     ↓
MobileNet Classifier
     ↓
ASL Letter + Probability Output
```

---

## Dataset

* **American Sign Language (ASL) Dataset** - https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az
* Contains labeled hand gesture images for alphabet classification


---

## Model Details

* **Base Model:** MobileNet
* **Reason:** Lightweight, fast inference, suitable for real-time applications
* **Output:** 26 ASL alphabets
* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** Adam

---

## Output Example

* Detected Letter: **A**
* Confidence: **92.4%**
* Displayed above the hand bounding box in real time

---

## Installation

```bash
pip install mediapipe opencv-python tensorflow numpy
```

---



## Results

Results

The proposed hand gesture to text conversion system demonstrates strong performance in real-time ASL alphabet recognition.

* Overall Accuracy: 96%
* Precision: 0.93
* Recall: 0.95

## Limitations

* Performance depends on lighting conditions
* Works best for single-hand gestures
* Background clutter may affect detection accuracy

---



