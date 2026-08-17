# 🩺 Explainable Multi-Disease Clinical Decision Support System

An AI-powered healthcare decision support system that predicts the risk of **Diabetes**, **Heart Disease**, and **Chronic Kidney Disease (CKD)** using **Graph Neural Networks (GNNs)** and provides explainable insights for each prediction.

---

## 🚀 Features

* Multi-disease risk prediction
* Graph Neural Network-based learning
* Explainable AI (XAI) integration
* Interactive clinical dashboard
* Patient risk assessment reports
* Real-time prediction through web interface
* Clinical feature importance visualization

---

## 🎯 Diseases Covered

* 🩸 Diabetes
* ❤️ Heart Disease
* 🫘 Chronic Kidney Disease (CKD)

---

## 📊 Dataset

This project utilizes data derived from the **NHANES (National Health and Nutrition Examination Survey)** dataset.

### Clinical Variables

* Age
* BMI
* Waist Circumference
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Hypertension
* HDL Cholesterol
* Total Cholesterol
* Glucose
* HbA1c
* Creatinine
* BUN

---

## 🧠 Model Architecture

```text
Clinical Data
      │
      ▼
Data Preprocessing
      │
      ▼
Graph Construction
      │
      ▼
Graph Neural Network
      │
 ┌────┼────┐
 ▼    ▼    ▼
DM   HD   CKD
      │
      ▼
Explainable AI
      │
      ▼
Risk Report
```

---

## 🛠️ Tech Stack

### Frontend

* React.js
* HTML5
* CSS3
* JavaScript

### Backend

* FastAPI
* Python

### Machine Learning

* PyTorch
* PyTorch Geometric
* Scikit-learn
* Pandas
* NumPy

### Explainable AI

* GNNExplainer
* SHAP

### Visualization

* Matplotlib
* Seaborn

### Database

* PostgreSQL

---

## 📂 Project Structure

```bash
project/
│
├── frontend/
├── backend/
├── data/
├── models/
├── notebooks/
├── explainability/
├── docs/
│
├── requirements.txt
└── README.md
```

---

## 🔬 Workflow

```text
NHANES Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Graph Construction
      │
      ▼
GNN Training
      │
      ▼
Multi-Disease Prediction
      │
      ▼
Explainability
      │
      ▼
Web Dashboard
```

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

## 🌐 Web Application Modules

### Landing Page

Project introduction and overview.

### Patient Input Form

Clinical data entry interface.

### Prediction Dashboard

Displays disease risk scores.

### Explainability Dashboard

Shows factors influencing predictions.

### Clinical Summary

Generates an easy-to-understand patient report.

---

## 🔮 Future Enhancements

* Additional disease prediction modules
* Advanced Graph Attention Networks (GAT)
* Electronic Health Record (EHR) integration
* Cloud deployment
* Personalized health recommendations

---

## 👨‍💻 Team Project

**Project Title:**
**Explainable Multi-Disease Clinical Decision Support System Using Graph Neural Networks**

**Domain:** Artificial Intelligence in Healthcare

**Focus Areas:**

* Graph Machine Learning
* Clinical Decision Support Systems
* Explainable AI
* Healthcare Analytics
