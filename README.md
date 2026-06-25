# 🚀 Advanced Machine Learning Portfolio (CodeAlpha Tasks)

Welcome to my machine learning task repository. This project showcases three distinct end-to-end applications ranging from predictive risk analytics to deep learning computer vision and automated clinical screening. 

Each subfolder contains a decoupled pipeline separating the core model optimizations (`train.py`) from interactive web dashboards built using the **Streamlit** ecosystem (`app.py`).

---

## 📂 Repository Blueprint & Subprojects

### 1. 💳 Credit Scoring Engine (`credit_scoring_project`)
An enterprise-grade underwriting assessment tool engineered to calculate individual default probabilities over a 2-year horizon using historical banking profiles.
* **Core Architecture:** Balanced Ensemble Random Forest Topology
* **Validation Score:** **0.8346 ROC-AUC** | **71% Defaulter Recall**
* **Advanced Metrics:** Integrated 45% Debt-to-Income (DTI) maximum capital cap tracking, dynamic Prime/Subprime tiering, and automated adverse risk reason code logs.

### ✍️ 2. Handwritten Digit Recognition (`handwritten_prediction`)
A Computer Vision application that interprets freehand numerical drawings in real-time through an interactive interface.
* **Core Architecture:** Deep Convolutional Neural Network (CNN) optimized via TensorFlow/Keras
* **Validation Score:** **99.2% Test Accuracy** on the MNIST framework
* **Advanced Metrics:** Built-in OpenCV preprocessing layers, automatic grayscale downsampling ($28 \times 28$ matrix scaling), and continuous feature array confidence charts.

### 🩺 3. Multi-Variant Disease Predictor (`Disease_prediction`)
A diagnostic triage assistant designed to map variable systemic symptoms to localized medical condition outcomes.
* **Core Architecture:** Multi-Class Support Vector Machine (SVM) utilizing a Linear/RBF classification boundary
* **Validation Score:** **96.8% Global F1-Score Accuracy**
* **Advanced Metrics:** Structured multi-symptom tokenization matrices and multi-tiered clinical priority alert configurations.

---

## ⚙️ Local Deployment & Environment Setup

To run any of the interactive web dashboards locally on your machine, follow these standard installation routines:

### Step 1: Clone the Code Base
```bash
git clone [https://github.com/afridklassic-01/Codealpha_task.git](https://github.com/afridklassic-01/Codealpha_task.git)
cd Codealpha_task
