# Current Project Audit Report (Updated Post-Audit)

**Date:** 2026-09-20  
**Project:** Explainable Multi-Disease Clinical Decision Support System Using Graph Neural Networks  
**Scope:** Academic ML System Audit (Phase 0 through Phase 19)

---

## 1. Executive Summary & Audit Questions

| # | Question | Findings & Post-Audit Verification |
|---|---|---|
| **1** | **Which datasets actually exist locally?** | `NHANES_diabetes_cleaned.csv` (11,452 rows, 581 KB) and `NHANES_heart_disease_cleaned.csv` (7,770 rows, 439 KB) in the workspace root. Also small summary CSVs in `reports/eda/`. |
| **2** | **Which datasets are only referenced in code?** | `NHANES_CKD_cleaned.csv` (planned for CKD), PTB-XL database (`data/ptbxl/ptbxl_database.csv`), PTB-XL raw waveforms (`records100/`, `records500/`), and medical image datasets. |
| **3** | **Does PTB-XL actually exist locally?** | **NO.** `data/ptbxl/` and raw waveform folders do not exist in the local workspace. Can be acquired via `python scripts/download_ptbxl.py`. |
| **4** | **Does NHANES actually exist locally?** | **YES.** Both diabetes and heart disease cleaned CSVs are present. |
| **5** | **Are real ECG files present?** | **NO.** No `.dat` or `.hea` waveform signal files exist locally. |
| **6** | **Are real image files present?** | **NO.** Only UI assets (`hero.png`) and generated EDA visualization plots exist. No medical images exist. |
| **7** | **Does a trained model checkpoint exist?** | **NO.** `models/ptbxl_multimodal/best_model.pt` does not exist on disk (ignored by `.gitignore`). |
| **8** | **Does a fitted preprocessing/scaler artifact exist?** | **NO.** `models/ptbxl_multimodal/preprocessor.joblib` does not exist on disk (ignored by `.gitignore`). Default fallback standardizer is used. |
| **9** | **Which models have actually been trained?** | **None on this local machine.** Training reports in `reports/` reference runs conducted on a collaborator's machine. Training routine verified on CPU. |
| **10** | **Which files are only architecture / placeholders?** | `app/models/image_encoder.py`, `app/preprocessing/image_preprocessor.py`, `graph/*.py` (legacy graph scaffold), `explainability/*.py` (empty 0-byte scripts). |
| **11** | **Which tests exist?** | 10 test suites in `tests/`: `test_clinical_preprocessor.py`, `test_ecg_encoder.py`, `test_encoders.py`, `test_image_preprocessor.py`, `test_patient_graph.py`, `test_ptbxl_preprocessor.py`, `test_eda_ptbxl.py`, `test_backend_api.py`, `test_ptbxl_inference.py`, `test_ptbxl_multimodal_training.py`. |
| **12** | **Which tests actually run?** | **ALL 37 tests across all 10 suites now run and pass (100% pass rate).** |
| **13** | **Which parts of the frontend call the backend?** | `frontend/src/services/api.js` calls `POST /predict`, `GET /health`, and `GET /model-info`. Triggered by `PatientReview.jsx` upon clicking "Run Assessment". |
| **14** | **Which backend endpoint loads which model?** | `POST /predict` in `backend/app/api/routes.py` calls `ClinicalPredictionService` -> `PTBXLInferenceService`, which attempts to load `PTBXLMultimodalGNN` from `models/ptbxl_multimodal/best_model.pt`. |
| **15** | **What exact features enter each trained model?** | `PTBXLMultimodalGNN` takes: 4 demographic features (`['age', 'sex', 'height', 'weight']`) + 12-channel ECG signal `(12, 1000)`. The other 10 lab metrics collected in the frontend form are bypassed during neural inference and used for heuristic clinical attribution. |

---

## 2. Status Categorization

### A. VERIFIED COMPLETE
- **Automated Test Suite**: 37 tests passing across all 10 test files.
- **Python Environment**: `wfdb`, `torch`, `torchvision`, `fastapi`, `uvicorn`, `scikit-learn`, `seaborn`, `httpx` verified. `requirements.txt` created.
- **Tabular NHANES Data**: `NHANES_diabetes_cleaned.csv` and `NHANES_heart_disease_cleaned.csv` loaded and verified.
- **PyTorch Neural Encoders (Modules)**:
  - `ClinicalEncoder` (`app/models/clinical_encoder.py`): Verified dimension transformation (4 -> [128, 64] -> 64).
  - `ECGEncoder` (`app/models/ecg_encoder.py`): Verified 1D CNN waveform feature extraction (12 leads -> 128-dim).
  - `MultimodalFusion` (`app/models/multimodal_model.py`): Verified fused multimodal embeddings.
- **Patient Similarity Graph**:
  - `PatientGraphBuilder` (`app/graph/patient_graph.py`): Builds $k$-NN graph using cosine similarity without leaking labels.
- **Pure PyTorch Message Passing**:
  - `MultiDiseaseGNN` (`app/models/gnn_model.py`): Functional PyTorch message-passing backbone (`SimpleGNNConv`) with multi-head logits.
- **EDA & Leakage Verification Suites**:
  - `scripts/eda_ptbxl.py` and `tests/test_ptbxl_preprocessor.py` verify 3-way disjoint patient splitting logic (Train $\cap$ Val = $\emptyset$, Train $\cap$ Test = $\emptyset$, Val $\cap$ Test = $\emptyset$).
- **API & Portability**:
  - Hardcoded absolute Windows paths replaced with `PROJECT_ROOT` relative paths across scripts and tests.
  - Pydantic v2 `ConfigDict` updated in `backend/app/api/schemas.py`.
  - Clean error handling (HTTP 503) when checkpoint is absent.
- **Frontend User Interface**:
  - React 19 + Vite dashboard, clinical intake forms, review flows, and visualizations.
  - Production build (`npm run build`) builds cleanly in 1.83s.

### B. IMPLEMENTED BUT NOT VERIFIED ON CURRENT MACHINE
- **Full PTB-XL Model Training**:
  - `ml_pipeline/train_ptbxl_multimodal.py`: Code and training step verified on synthetic batch, but full 15-epoch training requires downloading raw waveform data (~1.8 GB).

### C. MISSING
- **Checkpoints**: `models/ptbxl_multimodal/best_model.pt` and `preprocessor.joblib`.
- **Raw Datasets**: `data/ptbxl/`, `records100/`, `records500/`, `NHANES_CKD_cleaned.csv`.
- **Real Model-Derived XAI**: The current XAI feature attribution in `prediction_service.py` is heuristic rule-based rather than model gradients (e.g. Integrated Gradients, SHAP, or GNNExplainer).

### D. SCIENTIFIC & ARCHITECTURAL RISKS
1. **Clinical Feature Disconnect**: The frontend intake form presents a 14-variable cardiometabolic panel, but the PTB-XL model only consumes 4 demographic features (`age`, `sex`, `height`, `weight`) + 12-lead ECG. The other 10 lab values are currently not ingested by the neural network. (Documented in `reports/MODEL_FEATURE_MAPPING.md`).
2. **Dataset Partitioning Assumptions**: NHANES (survey cohort) and PTB-XL (ECG clinical cohort) are independent datasets. They must never be artificially matched across patient records.
3. **Missing Model Handling**: When a trained checkpoint is not available on disk, the system cleanly reports `503 Trained checkpoint not available` rather than fabricating predictions.
