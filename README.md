# Brain Tumor Classification using CNN

## Overview
This project classifies brain MRI scans into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The model uses Transfer Learning with TensorFlow/Keras and is deployed through a Streamlit web application.

---

## Dataset

Dataset: Brain MRI Images Dataset

Total Images: XXXX

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Model Architecture

Base Model:
- EfficientNetB0

Additional Layers:
- Global Average Pooling
- Dropout (0.3)
- Dense Softmax Layer

---

## Performance

| Metric | Value |
|----------|----------|
| Accuracy | 98.1% |
| Precision | 97.8% |
| Recall | 97.6% |
| F1 Score | 97.7% |

---

## Features

- MRI image upload
- Tumor classification
- Confidence score
- Real-time prediction
- Streamlit web interface

---

## Screenshots

(Add screenshots here)

---

## Project Structure

brain-tumor-classification/
│
├── app.py
├── train.py
├── model/
│   └── model.h5
├── dataset/
├── requirements.txt
└── README.md

---

## Installation

pip install -r requirements.txt

streamlit run app.py
