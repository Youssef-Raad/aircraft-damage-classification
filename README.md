# Aircraft Damage Detection & Captioning

A deep learning computer vision project built with TensorFlow/Keras and Hugging Face Transformers to classify aircraft surface defects (**dents** vs. **cracks**) and automatically generate natural-language inspection captions and summaries.

---

## Project Overview

Automated visual inspection is critical in aviation maintenance to identify and document structural damage early and reliably. This project combines two vision workflows in a single baseline pipeline:
1. **Defect Classification:** Binary image classification using a pre-trained **VGG16** backbone.
2. **Automated Image Captioning & Summarization:** Generative vision-language reporting using **BLIP** (`BlipForConditionalGeneration`).

---

## Dataset

The classification dataset consists of RGB aircraft surface images resized to **$224 \times 224 \times 3$** and normalized to $[0, 1]$ across two classes (`dent` and `crack`):

| Split | Number of Images | Purpose |
| :--- | :---: | :--- |
| **Train** | 300 | Model training |
| **Validation** | 96 | Epoch monitoring |
| **Test** | 50 | Final evaluation |
| **Total** | **446** | — |

---

## Model Architecture

### 1. Classification Model (VGG16 Transfer Learning)
* **Base Model:** VGG16 (`224x224x3` input, frozen ImageNet weights)
* **Classification Head:** `Flatten()` $\rightarrow$ `Dense(512, activation='relu')` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(512, activation='relu')` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(1, activation='sigmoid')`
* **Optimizer & Loss:** `Adam(learning_rate=0.0001)` with `binary_crossentropy`

### 2. Vision-Language Captioning (BLIP)
* **Model:** Pre-trained **BLIP** (`Salesforce/blip-image-captioning-base`)
* **Output:** Generates both a general scene caption and a targeted aircraft damage summary from input inspection images.

---

## Baseline Results

Classification model trained for **5 epochs** (batch size `32`):

| Metric | Score |
| :--- | :---: |
| **Training Accuracy** | 87.00% |
| **Validation Accuracy** | 69.79% |
| **Test Accuracy** | 68.75% |

---

## Project Structure

```text
aircraft-damage-classification/
├── Aircraft damage detection model.ipynb   # Baseline classification & BLIP captioning notebook
└── README.md                               # Project documentation
