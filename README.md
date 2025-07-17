# 🩺 Breast Cancer Prediction Using Machine Learning

This project predicts whether a breast tumor is **benign** or **malignant** using a **logistic regression classifier** trained on real diagnostic data.  
The entire workflow was built and tested on **Google Colab** using Python libraries like **Pandas** and **scikit-learn**.

---

## 📂 Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — available on **Kaggle**.
- **Features:** 30 numeric features describing characteristics of cell nuclei present in breast mass images.
- **Target:** Diagnosis — `M` (Malignant) or `B` (Benign).

---

## ✅ Problem Statement

> **Goal:** Develop a binary classification model that can accurately predict the presence of breast cancer based on diagnostic measurements.

---

## ⚙️ Tools & Libraries Used

- **Google Colab** — for interactive Python notebooks and GPU/CPU compute.
- **Pandas** — for data cleaning, exploration, and manipulation.
- **scikit-learn (sklearn)** — for logistic regression and model evaluation.
- **NumPy** — for numerical operations.
- **Matplotlib** and **Seaborn** — for data visualization.

---

## 🚀 How to Run

1️⃣ **Open in Google Colab**

Clone this repo or upload the notebook to your own Google Colab account.

2️⃣ **Install Required Libraries** (Colab usually has these pre-installed)
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
