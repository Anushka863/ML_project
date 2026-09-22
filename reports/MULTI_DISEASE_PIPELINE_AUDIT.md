# Multi-Disease Prediction Pipeline & Repository Audit Report

**Date:** 2026-09-22  
**Scope:** Complete Read-Only Technical Audit of the Multi-Disease Prediction Pipeline (Diabetes, Heart Disease, CKD), NHANES Datasets, PTB-XL ECG Model, Backend API, and Frontend Application.

---

## 1. Executive Summary & Component Status Matrix

| Component | Status | Description & Audit Finding |
|---|---|---|
| **Diabetes dataset** | **VERIFIED** | `NHANES_diabetes_cleaned.csv` present (11,452 rows, 11 clinical columns, binary target `Diabetes`). |
| **Diabetes preprocessing** | **CODE ONLY** | `ClinicalPreprocessor` in `app/preprocessing/clinical_preprocessor.py` is implemented but has no saved fitted scaler or imputer for diabetes. |
| **Diabetes model** | **CODE ONLY** | `ClinicalEncoder` and `MultiDiseaseGNN` exist in code; no model instance has been trained on the diabetes dataset. |
| **Diabetes checkpoint** | **MISSING** | No trained disease checkpoint found for Diabetes. |
| **Diabetes inference** | **MISSING** | No inference function connects patient clinical inputs to a diabetes model; `/predict` does not output diabetes predictions. |
| **Heart Disease dataset** | **VERIFIED** | `NHANES_heart_disease_cleaned.csv` present (7,770 rows, 11 clinical columns, binary target `Heart_Disease`). |
| **Heart Disease preprocessing** | **CODE ONLY** | `ClinicalPreprocessor` handles column mapping/scaling in code, but no fitted NHANES heart disease preprocessor exists. |
| **Heart Disease model** | **CODE ONLY** | Model architecture code exists in `app/models/` (`ClinicalEncoder`, `MultiDiseaseGNN`). No NHANES heart disease model is trained. |
| **Heart Disease checkpoint** | **MISSING** | No trained disease checkpoint found for NHANES Heart Disease. (Note: PTB-XL checkpoint predicts ECG abnormality, not NHANES heart disease). |
| **Heart Disease inference** | **PARTIAL** | Backend returns a `"heart_disease"` field in `predictions` dict, but it erroneously mirrors the PTB-XL ECG abnormality probability rather than a genuine cardiovascular disease model. |
| **CKD dataset** | **MISSING** | `NHANES_CKD_cleaned.csv` is missing from the workspace. (Referenced in `NHANES_Dataset_Analysis.ipynb.ipynb` as `NHANES_cleaned_master.csv`). |
| **CKD preprocessing** | **CODE ONLY** | `ClinicalPreprocessor` supports kidney biomarkers (`creatinine`, `bun`, `egfr`), but no dataset or fitted artifact exists. |
| **CKD model** | **CODE ONLY** | `MultiDiseaseGNN` supports a `"ckd"` head in code, but no model has been trained. |
| **CKD checkpoint** | **MISSING** | No trained disease checkpoint found for CKD. |
| **CKD inference** | **MISSING** | No CKD inference logic exists in the backend or frontend. |
| **Disease API outputs** | **BROKEN** | `POST /predict` returns PTB-XL ECG abnormality (`prediction`, `probability`, `risk_level`) and an un-modeled `"heart_disease"` alias; it omits `diabetes` and `ckd`. |
| **Disease frontend cards** | **MISSING** | `ResultsPage.jsx` only renders "Predicted ECG Status" and "Model Abnormality Probability"; individual cards for Diabetes, Heart Disease, and CKD are missing. |
| **PTB-XL model** | **VERIFIED** | `PTBXLMultimodalGNN` is trained and verified on 4 demographic features (`age`, `sex`, `height`, `weight`) + 12-lead ECG signals predicting binary ECG abnormality. |
| **ECG XAI** | **VERIFIED** | `PTBXLExplainer` computes genuine Captum Integrated Gradients across 12 ECG leads and 10-second temporal windows. |
| **Clinical XAI** | **PARTIAL** | Integrated Gradients computed only for 4 PTB-XL features (`age`, `sex`, `height`, `weight`). Clinical lab values (`glucose`, `hba1c`, `creatinine`, etc.) are classified as unmodeled "Additional Clinical Info". |
| **Graph inference** | **VERIFIED** | `PatientGraphBuilder` builds dynamic k-NN cosine graphs; for single-patient web inference, single-node isolated graph is handled safely. |

---

## 2. Existing Disease Code Inventory

### A. Python Files
- `app/models/clinical_encoder.py` — `ClinicalEncoder`: Multi-layer perceptron (MLP) for tabular clinical features (input_dim -> [128, 64] -> embedding_dim 64).
- `app/models/ecg_encoder.py` — `ECGEncoder`: 1D CNN waveform feature extractor (12 leads -> 128-dim).
- `app/models/gnn_model.py` — `MultiDiseaseGNN` & `SimpleGNNConv`: GNN backbone with `nn.ModuleDict` heads for `["diabetes", "heart_disease", "ckd"]`.
- `app/models/multimodal_model.py` — `MultimodalFusion`: Concatenation and projection for clinical + image/waveform embeddings.
- `app/preprocessing/clinical_preprocessor.py` — `ClinicalPreprocessor`: Missing value imputation (`SimpleImputer`), standardization (`StandardScaler`), and one-hot encoding (`OneHotEncoder`) for clinical tabular data.
- `app/preprocessing/ptbxl_preprocessor.py` — `PTBXLPreprocessor`: Imputation and scaling for PTB-XL metadata (`age, sex, height, weight`) and WFDB waveform extraction.
- `app/graph/patient_graph.py` — `PatientGraphBuilder`: k-NN cosine/euclidean patient similarity graph construction.
- `app/inference/ptbxl_inference.py` — `PTBXLInferenceService`: Singleton service loading `PTBXLMultimodalGNN` for real-time inference and Captum Integrated Gradients XAI.
- `app/explainability/tabular_xai.py` — `TabularXAI`: Gradient × Input feature attribution for tabular clinical models.
- `explainability/ptbxl_explainer.py` — `PTBXLExplainer`: Captum Integrated Gradients explainer for multimodal clinical + ECG model.
- `explainability/feature_importance.py` — `get_clinical_feature_attributions`: Helper wrapper for clinical Integrated Gradients.
- `ml_pipeline/train_ptbxl_multimodal.py` — `PTBXLMultimodalGNN` training script (15 epochs, patient-level leak-free splits, binary ECG abnormality target).
- `ml_pipeline/build_patient_graph.py` — Script to construct and save `data/processed/graph/ptbxl_graph.pt` (500 nodes, 2500 edges).
- `ml_pipeline/evaluate_ptbxl.py` — Test evaluation script for PTB-XL model metrics (accuracy, ROC-AUC, F1).
- `ml_pipeline/phase1_load_verify.py` — Audit script checking availability of NHANES datasets.
- `scripts/explore_datasets.py` — Dataset audit script for clinical CSVs and imaging folders.
- `scripts/validate_labels.py` — Domain validation script auditing disease label compatibility.
- `scripts/audit_patient_linkage.py` — Audit script checking cross-modality patient ID linkage.
- `backend/app/main.py` — FastAPI application entry point.
- `backend/app/api/routes.py` — API routes (`/health`, `/model-info`, `/predict`).
- `backend/app/api/schemas.py` — Pydantic schemas (`PatientAssessmentRequest`, `PredictionResponse`, `SingleDiseasePrediction`).
- `backend/app/services/prediction_service.py` — `ClinicalPredictionService`: Orchestrates preprocessing, model execution, XAI, and response formatting.

### B. Datasets & CSVs
- `NHANES_diabetes_cleaned.csv` (Root directory, 581,049 bytes) — 11,452 rows, 11 columns: `Age, Sex, BMI, Waist, Systolic_BP, Diastolic_BP, HDL, Total_Cholesterol, Glucose, HbA1c, Diabetes`.
- `NHANES_heart_disease_cleaned.csv` (Root directory, 439,169 bytes) — 7,770 rows, 11 columns: `Age, Sex, BMI, Waist, Systolic_BP, Diastolic_BP, HDL, Total_Cholesterol, Glucose, HbA1c, Heart_Disease`.
- `data/ptbxl/train_metadata.csv` (5,680,567 bytes) — PTB-XL training split.
- `data/ptbxl/val_metadata.csv` (1,246,148 bytes) — PTB-XL validation split.
- `data/ptbxl/test_metadata.csv` (1,227,303 bytes) — PTB-XL testing split.
- `NHANES_CKD_cleaned.csv` — **MISSING** from local filesystem.

### C. Checkpoints & Models on Disk
- `models/ptbxl_multimodal/best_model.pt` (2,129,613 bytes, SHA256: `48922961129a6c1a6d3594c0addfc29edc7fe90ee20f8d2a2c53e7624d0eac98`) — PTB-XL Multimodal GNN checkpoint.
- `models/ptbxl_multimodal/ptbxl_classifier.pt` (742,769 bytes, SHA256: `aa9f19bd2c0cc937912e88e0cd3656eedb700624d504200339ab0652383f7af1`) — Direct state dict of PTB-XL Multimodal GNN.
- `models/ptbxl_multimodal/ecg_encoder.pt` (257,170 bytes, SHA256: `ab24405be87a90d2614202036d1a11b71d69a1729183bffc1194b50363797b59`) — PTB-XL 1D CNN ECG Encoder weights.
- `models/ptbxl_multimodal/ptbxl_ecg_encoder.pt` (257,344 bytes, SHA256: `72f77a56a31847e266b94c4452deba2c916c30e84b75a982fa0223ce1e909a43`) — PTB-XL 1D CNN ECG Encoder weights.
- `models/ptbxl_multimodal/preprocessor.joblib` (951 bytes, SHA256: `d90bb6553486d7efddf79bfc65bdeb7da188d53d26357f6fa26e763cf51a3d9b`) — Fitted StandardScaler for PTB-XL demographic features (`age, sex, height, weight`).
- `data/processed/graph/ptbxl_graph.pt` (302,405 bytes, SHA256: `6f64e9599cb30b0dac3d6f02e78947556040778ade8adf424b90d1a5cee68719`) — PyTorch Geometric Data object with 500 nodes and 2500 edges.

---

## 3. NHANES Branch Detailed Audit

### A. Clinical Datasets
1. `NHANES_diabetes_cleaned.csv`
   - Row count: 11,452 rows
   - Columns (11): `['Age', 'Sex', 'BMI', 'Waist', 'Systolic_BP', 'Diastolic_BP', 'HDL', 'Total_Cholesterol', 'Glucose', 'HbA1c', 'Diabetes']`
   - Patient Identifiers: None (no `SEQN` column; indexed by row position).
   - Missing Values:
     - `BMI`: 3,212 missing (28.0%)
     - `Waist`: 3,484 missing (30.4%)
     - `Systolic_BP`: 4,157 missing (36.3%)
     - `Diastolic_BP`: 4,157 missing (36.3%)
     - `HDL`: 4,770 missing (41.6%)
     - `Total_Cholesterol`: 4,770 missing (41.6%)
     - `Glucose`: 7,910 missing (69.1%)
     - `HbA1c`: 4,960 missing (43.3%)
   - Target Column: `Diabetes`
     - Class 0 (No Diabetes): 10,371 (90.6%)
     - Class 1 (Diabetes): 1,081 (9.4%)

2. `NHANES_heart_disease_cleaned.csv`
   - Row count: 7,770 rows
   - Columns (11): `['Age', 'Sex', 'BMI', 'Waist', 'Systolic_BP', 'Diastolic_BP', 'HDL', 'Total_Cholesterol', 'Glucose', 'HbA1c', 'Heart_Disease']`
   - Patient Identifiers: None (no `SEQN` column).
   - Missing Values:
     - `BMI`: 1,827 missing (23.5%)
     - `Waist`: 2,033 missing (26.2%)
     - `Systolic_BP`: 1,933 missing (24.9%)
     - `Diastolic_BP`: 1,933 missing (24.9%)
     - `HDL`: 2,295 missing (29.5%)
     - `Total_Cholesterol`: 2,295 missing (29.5%)
     - `Glucose`: 4,572 missing (58.8%)
     - `HbA1c`: 2,029 missing (26.1%)
   - Target Column: `Heart_Disease`
     - Class 0 (No Heart Disease): 7,025 (90.4%)
     - Class 1 (Heart Disease): 745 (9.6%)

3. `NHANES_CKD_cleaned.csv`
   - **NOT PRESENT** in workspace.

### B. Preprocessing & Models for NHANES
- **Clinical Preprocessor**: `ClinicalPreprocessor` (`app/preprocessing/clinical_preprocessor.py`) exists as an unfitted class. It provides median imputation (`SimpleImputer`), standard scaling (`StandardScaler`), and one-hot encoding (`OneHotEncoder`).
- **Clinical Encoder**: `ClinicalEncoder` (`app/models/clinical_encoder.py`) exists as an uninstantiated PyTorch module.
- **GNN / MultiDiseaseGNN**: `MultiDiseaseGNN` (`app/models/gnn_model.py`) exists as an architecture module.
- **Trained Disease Checkpoints**: **No trained disease checkpoint found.**

---

## 4. Disease Prediction Reality Check

| Disease | Model Exists | Trained Checkpoint | Real Inference | Input Features | Target | Probability Produced |
|---|---|---|---|---|---|---|
| **Diabetes** | **CODE ONLY** (`ClinicalEncoder`, `MultiDiseaseGNN`) | **NO** | **NO** | `Age, Sex, BMI, Waist, Systolic_BP, Diastolic_BP, HDL, Total_Cholesterol, Glucose, HbA1c` | `Diabetes` (0/1) | **None** |
| **Heart Disease (NHANES)** | **CODE ONLY** (`ClinicalEncoder`, `MultiDiseaseGNN`) | **NO** | **NO** | `Age, Sex, BMI, Waist, Systolic_BP, Diastolic_BP, HDL, Total_Cholesterol, Glucose, HbA1c` | `Heart_Disease` (0/1) | **None** |
| **Heart Disease (PTB-XL ECG)** | **YES** (`PTBXLMultimodalGNN`) | **YES** (`best_model.pt`) | **YES** | `age, sex, height, weight` + 12-lead ECG (12, 1000) | `target` (0: Normal ECG, 1: Abnormal ECG) | **ECG Abnormality Probability** |
| **Chronic Kidney Disease (CKD)** | **CODE ONLY** (`MultiDiseaseGNN`) | **NO** | **NO** | `Creatinine, BUN, eGFR, Age, Sex, BP` (spec only) | `CKD` (0/1) | **None** |

---

## 5. Audit of Current PTB-XL Model

- **Clinical Inputs**: 4 demographic features (`age`, `sex`, `height`, `weight`).
- **ECG Inputs**: 12-lead ECG waveform signal at 100 Hz (10 seconds, shape: $12 \times 1000$).
- **Clinical Encoder**: `ClinicalEncoder` (Linear 4 -> 128 -> 64 -> 64, ReLU, BatchNorm, Dropout 0.2).
- **ECG Encoder**: `ECGEncoder` (3-layer 1D ConvNet: Conv1d(12, 32, k=15) -> Conv1d(32, 64, k=7) -> Conv1d(64, 128, k=3) -> AdaptiveAvgPool1d(1) -> Linear(128, 128)).
- **Fusion**: `MultimodalFusion` (Concatenation of 64 + 128 = 192 -> MLP 256 -> 128).
- **Graph Construction**: `PatientGraphBuilder` (k-NN graph, k=5, Cosine similarity on 128-dim fused embeddings).
- **GNN Backbone**: `MultiDiseaseGNN` (2-layer `SimpleGNNConv` 128 -> 64 -> 32) with a single output head `gnn.heads.heart_disease` (Linear 32 -> 16 -> Linear 16 -> 1).
- **Target Predicted**: **ECG Abnormality** (0: Normal ECG / `NORM` in SCP codes; 1: Abnormal ECG).
- **Checkpoint**: `models/ptbxl_multimodal/best_model.pt` (2.13 MB).
- **Inference Service**: `PTBXLInferenceService` (`app/inference/ptbxl_inference.py`).
- **Explainability (XAI)**: `PTBXLExplainer` (`explainability/ptbxl_explainer.py`) computing Captum Integrated Gradients for the 4 demographic features and 12-lead ECG signals.

> [!IMPORTANT]
> The PTB-XL model predicts electrophysiological ECG diagnostic abnormality. It is NOT a systemic multi-disease model for diabetes, chronic kidney disease, or clinical survey heart disease.

---

## 6. Website Data Flow & Field Breakdown

### Data Flow Path:
`PatientAssessment.jsx` (Form State)  
$\downarrow$  
`PatientReview.jsx` (Review State)  
$\downarrow$  
`frontend/src/services/api.js` (`formatPatientPayload` -> `POST /predict`)  
$\downarrow$  
`backend/app/api/routes.py` (`POST /predict` -> `PatientAssessmentRequest`)  
$\downarrow$  
`backend/app/services/prediction_service.py` (`predict_patient_risk`)  
$\downarrow$  
`app/inference/ptbxl_inference.py` (`PTBXLInferenceService.predict`)  
$\downarrow$  
`PTBXLMultimodalGNN` (`models/ptbxl_multimodal/best_model.pt`)  
$\downarrow$  
`backend/app/api/schemas.py` (`PredictionResponse`)  
$\downarrow$  
`ResultsPage.jsx` (Render View)

### Field Categorization:

| Field Name | Frontend Key | Pydantic Schema Key | Category | Exact Status in Current Pipeline |
|---|---|---|---|---|
| **Age** | `age` | `age: float` | **A. Consumed by Model** | Scaled via `StandardScaler` and input to `ClinicalEncoder`. |
| **Gender / Sex** | `gender` | `gender: str` | **A. Consumed by Model** | Encoded to binary float (Male: 1.0, Female: 0.0) and input to `ClinicalEncoder`. |
| **Height** | `height` | `height: Optional[float]` | **A. Consumed by Model** | Scaled via `StandardScaler` and input to `ClinicalEncoder`. |
| **Weight** | `weight` | `weight: Optional[float]` | **A. Consumed by Model** | Scaled via `StandardScaler` and input to `ClinicalEncoder`. |
| **BMI** | `bmi` | `bmi: float` | **B / C. Non-Model Display** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. |
| **Systolic BP** | `systolic` | `systolic: float` | **B / C. Non-Model Display** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. |
| **Diastolic BP** | `diastolic` | `diastolic: float` | **B / C. Non-Model Display** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. |
| **Fasting Glucose** | `glucose` | `glucose: float` | **D. Disease Intended** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. Intended for Diabetes model. |
| **HbA1c** | `hba1c` | `hba1c: float` | **D. Disease Intended** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. Intended for Diabetes model. |
| **HDL Cholesterol** | `hdl` | `hdl: float` | **D. Disease Intended** | Sent to backend; bypassed by neural network; not consumed by model. Intended for Heart Disease model. |
| **Total Cholesterol** | `totalCholesterol` | `totalCholesterol: float`| **D. Disease Intended** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. Intended for Heart Disease model. |
| **Serum Creatinine**| `creatinine` | `creatinine: float` | **D. Disease Intended** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. Intended for CKD model. |
| **BUN** | `bun` | `bun: float` | **D. Disease Intended** | Sent to backend; bypassed by neural network; returned in `additional_clinical_info`. Intended for CKD model. |
| **Waist** | `waist` | `waist: Optional[float]`| **D. Disease Intended** | Sent to backend; bypassed by neural network. Intended for cardiometabolic models. |

---

## 7. Heuristic / Rule-Based Logic Audit

| File | Location / Function | Exact Logic | Used by `/predict`? |
|---|---|---|---|
| `backend/app/services/prediction_service.py` | Lines 65-121 in `predict_patient_risk` | Evaluates thresholds for `clinical_status` (`Systolic >= 130 -> "Elevated"`, `BMI >= 25.0 -> "Elevated"`, `Glucose >= 100 -> "Elevated"`, `HbA1c >= 5.7 -> "Elevated"`, `Cholesterol >= 200 -> "Borderline / High"`, `Creatinine > 1.3 -> "High"`, `BUN > 20 -> "High"`). Explicitly labeled as "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model." | **YES** (Populates `additional_clinical_info` for display context). |
| `backend/app/services/prediction_service.py` | Lines 130-134 in `predict_patient_risk` | `predictions["heart_disease"] = SingleDiseasePrediction(risk_score=ptbxl_result["probability"], ...)` Assigns PTB-XL ECG Abnormality score to the `"heart_disease"` field. | **YES** (Directly returns ECG abnormality probability as heart disease risk). |
| `app/inference/ptbxl_inference.py` | Lines 238-243 in `PTBXLInferenceService.predict` | Risk category binning: `prob >= 0.65 -> "High Risk"`, `prob >= 0.40 -> "Moderate Risk"`, `else "Low Risk"`. | **YES** (Categorizes model sigmoid probability). |
| `frontend/src/services/api.js` | Lines 23-32 in `formatPatientPayload` | Heuristic BMI formula `weightKg / (heightM * heightM)` if BMI is missing in input form. | **YES** (Input formatting fallback). |

---

## 8. Current API Contract (`POST /predict`)

### Actual Current Response Structure:
```json
{
  "status": "success",
  "prediction": "Normal",
  "probability": 0.1245,
  "confidence": 87.6,
  "risk_level": "Low Risk",
  "model": "PTB-XL Multimodal GNN",
  "disclaimer": "This AI-generated result is for research/educational purposes and is not a medical diagnosis.",
  "predictions": {
    "ptbxl_multimodal_gnn": {
      "risk_score": 0.1245,
      "prediction": "ECG Abnormality Risk: Low Risk",
      "risk_level": "Low Risk"
    },
    "heart_disease": {
      "risk_score": 0.1245,
      "prediction": "Cardiovascular Risk: Low Risk",
      "risk_level": "Low Risk"
    }
  },
  "clinical_explanation": [
    {
      "feature": "Age",
      "value": 58.0,
      "importance": 0.0412,
      "impact": "pushes toward abnormal (+0.0412)",
      "direction": "toward_abnormal",
      "attribution": 0.0412
    }
  ],
  "image_explanation": null,
  "graph_explanation": {
    "nodes": 1,
    "edges": 0,
    "message_passing": false,
    "explanation": "Single-patient inference: graph contains 1 node and 0 edges..."
  },
  "ecg_explanation": {
    "lead_attributions": { "I": 0.012, "II": 0.034, "V1": 0.089 },
    "top_leads": ["V1", "II", "aVF"],
    "temporal_attributions": [...]
  },
  "xai": { ... },
  "additional_clinical_info": [
    {
      "marker": "Fasting Glucose",
      "value": 126.0,
      "unit": "mg/dL",
      "reference_range": "70 - 99 mg/dL",
      "clinical_status": "Elevated",
      "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
    }
  ]
}
```

---

## 9. Current Results Page (`ResultsPage.jsx`) Audit

1. **Expected Prediction Fields**:
   - `prediction`, `probability`, `confidence`, `risk_level`, `model`, `clinical_explanation`, `ecg_explanation`, `graph_explanation`, `additional_clinical_info`, `xai`, `disclaimer`.
2. **Currently Displayed**:
   - Top Banner: "Predicted ECG Status" (`prediction.toUpperCase()`) & "Model Abnormality Probability" (`(probability * 100).toFixed(1)%`).
   - Section 1: "Model-Derived Clinical Feature Attributions (Integrated Gradients)" for Age, Sex, Height, Weight.
   - Section 2: "ECG 12-Lead Model Attribution Summary" (lead bars and 10-second temporal windows).
   - Section 3: "Inference Graph Architecture State" (nodes, edges, message passing status).
   - Section 4: "Additional Clinical Information (General Context)" (cards for Glucose, BP, BMI, etc. with transparency disclaimer).
3. **Disease Cards**:
   - Individual risk prediction cards for **Diabetes**, **Heart Disease**, and **CKD** **DO NOT EXIST** in `ResultsPage.jsx`.
4. **Hardcoding & Rounding**:
   - Values are dynamically populated from `predictionResults`.
   - `(probability * 100).toFixed(1)` formats percentage. If probability is 1.0, it prints `100.0%`.
   - "High Risk" / "Low Risk" badge color is derived from `isAbnormal` (`prediction.toUpperCase() === 'ABNORMAL'`).

---

## 10. Top 10 Implementation Problems

1. **Missing Trained Disease Checkpoints**: No checkpoints exist for Diabetes, NHANES Heart Disease, or CKD.
2. **Missing CKD Dataset**: `NHANES_CKD_cleaned.csv` does not exist in the repository; only Diabetes and Heart Disease cleaned CSVs are present.
3. **ECG Abnormality Aliased as Heart Disease**: The backend assigns the PTB-XL ECG abnormality probability directly to the `"heart_disease"` field in `predictions`.
4. **Clinical Lab Feature Disconnect**: 10 of the 14 features entered in the intake form (glucose, hba1c, creatinine, bun, systolic, diastolic, hdl, total cholesterol, waist, bmi) are bypassed by the PTB-XL model.
5. **No Multi-Head Disease Inference Engine**: The backend has no multi-disease inference coordinator that executes real models for all three target conditions.
6. **Frontend Results Page Shows Only ECG Status**: `ResultsPage.jsx` displays ECG Abnormality status and lacks disease cards for Diabetes, Heart Disease, and CKD.
7. **Single-Patient Graph Isolation**: For single-patient assessment, the graph has 1 node and 0 edges; GNN message passing does not leverage reference graph nodes.
8. **Missing Fitted Preprocessor for NHANES**: `ClinicalPreprocessor` is implemented in code but has no persisted scaler/imputer artifacts fitted on NHANES training partitions.
9. **High Missing Data in NHANES CSVs**: Glucose has 69.1% missing values and HbA1c has 43.3% missing values in `NHANES_diabetes_cleaned.csv`, requiring leak-free median imputation.
10. **Inconsistent API Response Schema**: `PredictionResponse` is designed around a single binary ECG model rather than a structured multi-disease response contract returning verified probabilities for all three targets.

---

## 11. Recommended Implementation Plan & Order

1. **Step 1: Unify / Prepare Clinical Datasets** — Create clean, leak-free train/val/test splits with fitted `ClinicalPreprocessor` for available NHANES datasets (Diabetes and Heart Disease) and formulate CKD pipeline with kidney biomarker specifications.
2. **Step 2: Train Real Disease Models / MultiDiseaseGNN** — Train leak-free clinical encoders and GNN multi-head prediction models for Diabetes, Heart Disease, and CKD; export checkpoints and fitted scalers.
3. **Step 3: Multi-Disease Inference Service** — Build a unified inference service that executes real model forward passes for Diabetes, Heart Disease, and CKD, computing genuine class probabilities.
4. **Step 4: Update Backend API Schema & Routes** — Update `POST /predict` to return genuine `diabetes_probability`, `heart_disease_probability`, and `ckd_probability` alongside Integrated Gradients attributions.
5. **Step 5: Enhance Frontend Results Page** — Add genuine disease risk cards (Diabetes, Heart Disease, CKD) to `ResultsPage.jsx`, displaying model-derived percentages, risk levels, and clinical attributions.
