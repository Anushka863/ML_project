# 🩺 Explainable Multi-Disease Clinical Decision Support System

An AI-powered healthcare decision support system integrating **Graph Neural Networks (GNNs)**, **Multimodal Representation Learning**, and **Explainable AI (XAI)**.

---

## 🔬 System Architecture & Scientific Tracks

The project investigates clinical decision support across two complementary data tracks:

### Track A: NHANES Tabular Clinical System
- **Cohorts**: National Health and Nutrition Examination Survey (NHANES)
- **Datasets**: `NHANES_diabetes_cleaned.csv` (11,452 patients), `NHANES_heart_disease_cleaned.csv` (7,770 patients), `NHANES_CKD_cleaned.csv` (pending)
- **Features**: Age, Sex, BMI, Waist, Blood Pressure, HDL, Total Cholesterol, Glucose, HbA1c
- **Target**: Multi-disease risk screening for Diabetes, Heart Disease, and CKD

### Track B: PTB-XL Multimodal Electrocardiography GNN
- **Cohort**: PTB-XL Electrocardiography Dataset (21,799 12-lead ECG records from 18,869 patients)
- **Inputs**:
  - **Clinical Demographics ($x \in \mathbb{R}^4$)**: Age, Sex, Height, Weight
  - **ECG Waveforms ($x \in \mathbb{R}^{12 \times 1000}$)**: 12-lead 10-second potential waveforms sampled at 100 Hz
- **Encoders**:
  - `ClinicalEncoder`: Multi-layer Perceptron (4 -> [128, 64] -> 64-dim embedding)
  - `ECGEncoder`: 1D Convolutional Neural Network (12 leads -> 128-dim embedding)
  - `MultimodalFusion`: Joint representation fusion layer (128-dim fused embedding)
- **Graph & Message Passing**:
  - `PatientGraphBuilder`: Inductive $k$-NN graph ($k=5$, Cosine similarity)
  - `MultiDiseaseGNN`: PyTorch degree-normalized message-passing backbone (`SimpleGNNConv`) with disease classification heads
- **Target**: Binary diagnostic abnormality / cardiac pathology detection (`0 = NORM`, `1 = Abnormal`)

```text
Patient Clinical Data (Age, Sex, Height, Weight)   +   12-Lead ECG Waveform (10s @ 100Hz)
                    │                                                  │
                    ▼                                                  ▼
          Clinical MLP Encoder                               1D CNN Waveform Encoder
             (64-dim latent)                                     (128-dim latent)
                    │                                                  │
                    └─────────────────┬────────────────────────────────┘
                                      ▼
                          Multimodal Fusion Layer
                             (128-dim embedding)
                                      │
                                      ▼
                        Patient Similarity Graph
                         (k-NN, Cosine Metric)
                                      │
                                      ▼
                       Graph Neural Network (GNN)
                         (Message Passing Layers)
                                      │
                                      ▼
                    Cardiac Abnormality Risk Prediction
                                      │
                                      ▼
                      FastAPI Backend & React Frontend
```

> [!IMPORTANT]
> **Scientific Integrity Rules:**
> 1. NHANES and PTB-XL are distinct patient cohorts; they are never artificially cross-paired.
> 2. Multimodal fusion occurs strictly between clinical demographics and ECG signals belonging to the exact same PTB-XL patient.
> 3. Splitting is strictly patient-level disjoint (Train $\cap$ Val = $\emptyset$, Train $\cap$ Test = $\emptyset$, Val $\cap$ Test = $\emptyset$) to eliminate data leakage.

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch 2.13, Torchvision 0.28, Scikit-learn, Scipy
- **Bio-Signal Processing**: WFDB (Waveform Database)
- **Data & EDA**: Pandas, NumPy, Matplotlib, Seaborn
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard**: React 19, Vite, React Router, CSS3

---

## 🚀 Quick Start Guide

### 1. Python Environment Setup
```powershell
python -m pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```powershell
python -m pytest -q
```
*Current test suite: 37 tests across 10 modules (100% pass rate).*

### 3. Start FastAPI Backend
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Root: `http://127.0.0.1:8000`
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`
- Model Info: `http://127.0.0.1:8000/model-info`

### 4. Start React Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📂 Project Directory Structure

```text
ML_project/
├── app/
│   ├── graph/                  # Patient similarity graph builder (k-NN, cosine)
│   ├── inference/              # Production inference pipeline with safe eval mode
│   ├── models/                 # PyTorch encoders (Clinical MLP, 1D CNN, GNN)
│   ├── preprocessing/          # Leakage-free splitters & standardizers
│   └── services/               # Core application service classes
├── backend/
│   └── app/
│       ├── api/                # FastAPI routes and Pydantic v2 schemas
│       ├── services/           # Prediction service integration
│       └── main.py             # FastAPI entry point with CORS
├── frontend/
│   ├── src/
│   │   ├── components/         # Reusable UI cards, navbars, forms
│   │   ├── pages/              # Landing, HowItWorks, Assessment, Review, Results
│   │   └── services/           # API client (predictPatientRisk, checkHealth)
│   └── package.json
├── ml_pipeline/
│   ├── phase1_load_verify.py   # NHANES data inspection script
│   ├── train_ptbxl_multimodal.py # Multimodal GNN training loop
│   └── evaluate_ptbxl.py       # Hold-out test set evaluation script
├── reports/
│   ├── CURRENT_PROJECT_AUDIT.md # Comprehensive academic audit findings
│   ├── MODEL_FEATURE_MAPPING.md # Exact feature ingestion mapping
│   ├── PTBXL_PIPELINE_STATUS.md # Component status & reproduction guide
│   └── final_evaluation.md     # Documented test metrics
├── scripts/
│   ├── download_ptbxl.py       # Dataset downloader
│   └── eda_ptbxl.py            # Reproducible Exploratory Data Analysis
├── tests/                      # 10 automated test suites (37 test functions)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📊 Evaluation & Audit Documentation

For complete technical and academic details, refer to:
- [`reports/CURRENT_PROJECT_AUDIT.md`](file:///c:/Users/kashi/VSCode_Projects/Machine/ML_project/reports/CURRENT_PROJECT_AUDIT.md): Detailed inventory, gap analysis, and component status.
- [`reports/MODEL_FEATURE_MAPPING.md`](file:///c:/Users/kashi/VSCode_Projects/Machine/ML_project/reports/MODEL_FEATURE_MAPPING.md): Clinical form to model input feature traceability.
- [`reports/PTBXL_PIPELINE_STATUS.md`](file:///c:/Users/kashi/VSCode_Projects/Machine/ML_project/reports/PTBXL_PIPELINE_STATUS.md): Training instructions and execution status.
