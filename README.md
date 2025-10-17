# 🌆 UrbanTrace: Lifestyle & Pollution Risk Prediction

**UrbanTrace** is a data-driven system designed to analyze how urban lifestyle, commuting habits, and location-based factors influence pollution-related health risks.  
Using a **machine learning approach**, the system predicts pollution risk levels — **Low**, **Medium**, or **High** — based on lifestyle and environmental factors.

---

## 🚀 Overview

UrbanTrace integrates a **Random Forest Classifier** for pollution risk prediction, with:
- A **FastAPI backend** for model deployment.
- A **Streamlit frontend** for interactive user input and visualization.

---

## ✨ Features

- 🧠 Predicts pollution risk based on lifestyle and environmental parameters.  
- 💻 Interactive web interface for real-time predictions.  
- ⚙️ Comprehensive data preprocessing and feature engineering.  
- 📊 Model evaluation with accuracy, precision, recall, and confusion matrix.  
- 🔍 Composite indicators such as:
  - **Daily Exposure Index**
  - **Industry Proximity Score**

---

## 📚 Dataset

**Name:** UrbanTrace: Lifestyle & Pollution Insights  
**Records:** ~10,000  

### Features
- Commuting time  
- Vehicle type  
- Indoor air quality  
- Energy consumption  
- Proximity to industrial zones  

**Target:** Pollution risk category — *Low*, *Medium*, *High*

---

## 🧩 Methodology

### 🧹 Data Preprocessing
- Removed missing values and duplicates.  
- One-hot encoded categorical features.  
- Standardized continuous features using **z-score normalization**.  
- Managed outliers and inconsistencies to reduce skewness.

### 🏗️ Feature Engineering
- Conducted correlation analysis and feature importance evaluation.  
- Generated composite indicators like **Daily Exposure Index** and **Industry Proximity Score**.  
- Normalized all numerical features for consistency.

### 🤖 Model Development
- Implemented **Random Forest Classifier** for robust and interpretable predictions.  
- Split dataset into **80% training** and **20% testing**.  
- Performed hyperparameter tuning for optimal performance.

### 📈 Model Evaluation
- Evaluated using **confusion matrix**.  
- Computed **accuracy**, **precision**, and **recall** to assess model performance.

---

## 🛠️ Tools and Technologies

| Category | Tools |
|-----------|--------|
| **Programming Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |
| **Backend Framework** | FastAPI |
| **Frontend Framework** | Streamlit |

---

## 🌐 Deployment

- The **FastAPI backend** serves the trained model through REST API endpoints.  
- The **Streamlit frontend** allows users to input parameters and receive real-time predictions.  
- Ensures seamless communication between the model and the interactive interface.

---

## 🧭 How to Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
