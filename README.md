# 🌆 UrbanTrace: Lifestyle & Pollution Risk Prediction

**UrbanTrace** is a data-driven machine learning system designed to analyze how urban lifestyle, commuting habits, and environmental factors influence pollution-related health risks.
The system predicts pollution risk levels — **Low**, **Medium**, or **High** — using a machine learning pipeline with experiment tracking and model lifecycle management powered by **MLflow**.

---

## 🚀 Overview

UrbanTrace integrates a **Random Forest Classifier** for pollution risk prediction with:

* A **FastAPI backend** for model serving and API deployment.
* A **Streamlit frontend** for interactive predictions and visualization.
* **MLflow** for experiment tracking, model versioning, artifact logging, and reproducibility.

---

## ✨ Features

* 🧠 Predicts pollution risk using lifestyle and environmental parameters.
* 💻 Interactive web interface for real-time predictions.
* ⚙️ End-to-end preprocessing and feature engineering pipeline.
* 📊 Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
* 🔍 Composite indicators such as:

  * **Daily Exposure Index**
  * **Industry Proximity Score**
* 📁 MLflow-based experiment tracking and model registry support.
* 🔄 Dynamic model loading using MLflow Staging and Production environments.

---

## 📚 Dataset

**Name:** UrbanTrace: Lifestyle & Pollution Insights
**Records:** ~10,000

### Features

* Commuting time
* Vehicle type
* Indoor air quality
* Energy consumption
* Proximity to industrial zones

**Target:** Pollution risk category — *Low*, *Medium*, *High*

---

## 🧩 Methodology

### 🧹 Data Preprocessing

* Removed missing values and duplicates.
* One-hot encoded categorical features.
* Standardized numerical features using **StandardScaler**.
* Handled outliers and inconsistencies to improve data quality.

### 🏗️ Feature Engineering

* Performed correlation analysis and feature importance evaluation.
* Generated composite indicators like **Daily Exposure Index** and **Industry Proximity Score**.
* Normalized numerical features for consistency.

### 🤖 Model Development

* Implemented a **Random Forest Classifier** for robust predictions.
* Split dataset into **80% training** and **20% testing** sets.
* Performed hyperparameter tuning for improved model performance.

### 📈 Model Evaluation

* Evaluated using:

  * Confusion Matrix
  * Accuracy
  * Precision
  * Recall
  * F1-score

### 📦 MLflow Integration

* Tracked experiments, parameters, metrics, and artifacts using **MLflow**.
* Logged preprocessing artifacts, trained models, and evaluation reports.
* Used **MLflow Model Registry** for version control and stage management (*Staging* / *Production*).
* Enabled reproducible training pipelines and deployment decoupling through registry-based model loading.

---

## 🛠️ Tools and Technologies

| Category                    | Tools                                   |
| --------------------------- | --------------------------------------- |
| **Programming Language**    | Python                                  |
| **Libraries**               | Pandas, NumPy, Scikit-learn, Matplotlib |
| **Experiment Tracking**     | MLflow                                  |
| **Backend Framework**       | FastAPI                                 |
| **Frontend Framework**      | Streamlit                               |
| **Development Environment** | Jupyter Notebook                        |

---

## 🌐 Deployment

* The **FastAPI backend** serves the trained model through REST API endpoints.
* The **Streamlit frontend** provides an interactive interface for predictions and visualizations.
* **MLflow Registry** manages model versions and deployment stages.
* Supports seamless communication between the ML pipeline, APIs, and frontend application.

---

## 🧭 How to Run

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   ```
