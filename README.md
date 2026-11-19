# Aircraft Damage Classification (VGG16 Transfer Learning)

A deep learning computer vision project built with TensorFlow and Keras to classify aircraft surface defects into two categories: **dents** and **cracks**.

---

## Project Overview

Automated visual inspection is critical in aviation maintenance to identify structural damage early and reliably. This project implements an initial baseline binary image classifier leveraging a pre-trained **VGG16** convolutional backbone with a custom dense classification head.

---

## Dataset

The dataset consists of RGB aircraft surface images resized to **$224 \times 224 \times 3$** and normalized to $[0, 1]$ across two classes (`dent` and `crack`):

| Split | Number of Images | Purpose |
| :--- | :---: | :--- |
| **Train** | 300 | Model training |
| **Validation** | 96 | Hyperparameter & epoch monitoring |
| **Test** | 50 | Final model evaluation |
| **Total** | **446** | — |

---

## Model Architecture

The model uses **VGG16** (pre-trained on ImageNet, `include_top=False`) with frozen convolutional layers as a feature extractor, followed by a custom classification head:

* **Base Model:** VGG16 (`224x224x3` input, frozen weights)
* **Feature Flattening:** `Flatten()`
* **Fully Connected Layers:** `Dense(512, activation='relu')` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(512, activation='relu')` $\rightarrow$ `Dropout(0.3)`
* **Output Layer:** `Dense(1, activation='sigmoid')`
* **Optimizer & Loss:** `Adam(learning_rate=0.0001)` with `binary_crossentropy`

---

## Baseline Results

Trained for **5 epochs** (batch size `32`):

| Metric | Score |
| :--- | :---: |
| **Training Accuracy** | 87.00% |
| **Validation Accuracy** | 69.79% |
| **Test Accuracy** | 68.75% |

---

## Project Structure

```text
aircraft-damage-classification/
├── aircraft_damage_classification.ipynb   # Baseline training & evaluation notebook
└── README.md                              # Project documentation
