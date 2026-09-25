# Medical Image and Patient-Linkage Audit Report

**Date of Audit:** September 22, 2026  
**Auditor:** Antigravity Advanced Agentic AI Assistant  
**Repository:** `ML_PROJECT-Anushka_branch / ML_project`  
**Scope:** Dataset Integrity, Patient Identifiers, True Cross-Modal Linkage, Existing Code Architecture, and Scientific Readiness for Multimodal Fusion.

---

## Executive Summary

A comprehensive, ground-truth audit of all data directories, metadata files, source code modules, machine learning training scripts, backend endpoints, and frontend user interfaces was conducted across the entire repository.

### Key Audit Findings:
1. **Zero 2D Medical Image Datasets Present:** No medical imaging datasets (e.g., retinal fundus photography for diabetic retinopathy, kidney/renal ultrasound or CT scans, chest X-rays, MRI scans, or DICOM/NIfTI files) are present anywhere in the repository. The only `.png` files in the repository are 1 frontend UI hero graphic (`frontend/src/assets/hero.png`) and 16 diagnostic evaluation/EDA plots generated during offline analysis in `reports/`.
2. **Zero Patient Identifiers in NHANES Data:** The clinical datasets (`NHANES_diabetes_cleaned.csv`, `NHANES_heart_disease_cleaned.csv`, and `data/processed/clinical/*.csv`) contain **zero patient identifiers**, zero subject IDs, and zero survey sequence numbers (`SEQN`). All respondent IDs were stripped during upstream preprocessing.
3. **No Valid Linkage Between NHANES and Any Image/Signal Dataset:** There is **NO VALID PATIENT-LEVEL LINKAGE** between the NHANES clinical datasets and any external image or signal dataset.
4. **PTB-XL is an Electrophysiological 1D Waveform Dataset (Not 2D Images):** The repository contains the complete PTB-XL ECG dataset (`data/ptbxl/`, 21,799 records across 18,869 unique patients). PTB-XL contains authentic, same-patient linkage between clinical demographics (`age`, `sex`, `height`, `weight`) and 10-second 12-lead ECG time-series waveforms (`.dat`/`.hea` WFDB format), but it contains **no medical images**.
5. **Code Status:** Generic image preprocessing (`ImagePreprocessor`), image feature extraction (`ImageEncoder` using ResNet18/EfficientNet-B0), and image Grad-CAM explainability (`ImageGradCAM`) exist as standalone, unit-tested classes in `app/`, but are **NOT USED** in any training pipeline, inference service, or backend route because no paired medical image dataset exists in the repository.

---

## Step 1 — Exhaustive Repository Search for Image Datasets

An exhaustive filesystem scan was performed searching for all standard medical and computer vision image formats (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.dcm`, `.nii`, `.nii.gz`, `.mat`, `.dat`, `.hea`).

### Scan Results by File Extension:
- `.dcm`, `.nii`, `.nii.gz`, `.tif`, `.tiff`, `.bmp`, `.jpg`, `.jpeg`: **0 files found.**
- `.png`: **17 files found total**:
  - `frontend/src/assets/hero.png`: 1 UI illustration asset.
  - `reports/confusion_matrix.png`, `reports/roc_curve.png`, `reports/ptbxl_training_curves.png`: 3 evaluation plots.
  - `reports/figures/ecg_attribution_overlay.png`, `reports/figures/ecg_lead_attributions.png`: 2 XAI visualization figures.
  - `reports/eda/*.png`: 9 exploratory data analysis distribution charts.
  - `frontend/src/assets/react.svg`, `vite.svg`: 2 vector icons.
- `.dat` & `.hea` (PhysioNet WFDB 1D Time-Series): **43,598 `.dat` files and 43,598 `.hea` files** located in `data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` (21,799 100 Hz recordings and 21,799 500 Hz recordings).

### Detailed Dataset-by-Dataset Audit:

#### 1. Diabetic Retinopathy Image Dataset
- **Dataset Name:** None present
- **Folder / Path:** N/A
- **Number of Images:** 0
- **Image Formats:** None
- **Train / Val / Test Structure:** None
- **Class Names / Labels:** None
- **Annotation / Metadata Files:** None
- **Patient / Image / Study / Examination IDs:** None
- **Linkage Status:** **NOT PRESENT IN REPOSITORY**

#### 2. Kidney / Renal Image Dataset (Ultrasound / CT / Stone)
- **Dataset Name:** None present
- **Folder / Path:** N/A
- **Number of Images:** 0
- **Image Formats:** None
- **Train / Val / Test Structure:** None
- **Class Names / Labels:** None
- **Annotation / Metadata Files:** None
- **Patient / Image / Study / Examination IDs:** None
- **Linkage Status:** **NOT PRESENT IN REPOSITORY**

#### 3. Generic Binary Medical Image Dataset
- **Dataset Name:** None present (Mock tensors synthesized in-memory during unit tests)
- **Folder / Path:** N/A
- **Number of Images:** 0
- **Image Formats:** None
- **Train / Val / Test Structure:** None
- **Class Names / Labels:** None
- **Annotation / Metadata Files:** None
- **Patient / Image / Study / Examination IDs:** None
- **Linkage Status:** **NOT PRESENT IN REPOSITORY**

#### 4. PTB-XL ECG Dataset (1D Electrophysiological Waveforms)
- **Dataset Name:** PTB-XL: A Large Publicly Available Electrocardiography Dataset (v1.0.3)
- **Folder / Path:** `data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
- **Number of Signal Records:** 21,799 12-lead ECG records
- **Format:** PhysioNet WFDB 16-bit binary signal (`.dat`) + text header (`.hea`)
- **Train / Val / Test Structure:** Stratified 10-fold patient assignment; partitioned in `data/ptbxl/` into:
  - `train_metadata.csv`: 15,208 records (13,208 unique patients, folds 1–8)
  - `val_metadata.csv`: 3,319 records (2,830 unique patients, fold 9)
  - `test_metadata.csv`: 3,272 records (2,831 unique patients, fold 10)
  - Zero patient overlap across splits (`set(train) ∩ set(val) ∩ set(test) = ∅`).
- **Class Names & Labels:** 71 SCP-ECG diagnostic, form, and rhythm statements grouped into 5 superclasses:
  - `NORM` (Normal ECG: 9,514 statements)
  - `MI` (Myocardial Infarction: 5,469 statements)
  - `STTC` (ST/T Changes: 5,235 statements)
  - `CD` (Conduction Disturbance: 4,899 statements)
  - `HYP` (Hypertrophy: 2,317 statements)
  - Project Target Formulation: Binary Classification (`0 = NORM`, `1 = Abnormal / Diagnostic Finding`).
- **Annotation / Metadata Files:**
  - `ptbxl_database.csv` (21,799 rows × 28 columns)
  - `scp_statements.csv` (71 rows × 13 columns)
- **Identifiers Present:**
  - `patient_id`: Unique integer patient identifier (18,869 unique IDs).
  - `ecg_id`: Unique record identifier (21,799 unique IDs).
  - `filename_lr` / `filename_hr`: Relative filesystem path to the exact WFDB waveform record.
  - `recording_date`: Exact timestamp of recording.
- **Linkage Status:** **VALID SAME-PATIENT LINKAGE FOR CLINICAL DEMOGRAPHICS + 1D ECG WAVEFORM.** (No 2D medical images).

---

## Step 2 — Clinical Distinction of Target Labels

Clinical and machine learning integrity requires precise distinction between what an image/signal biomarker measures versus a systemic clinical diagnosis:

| Modality / Target | What the Data Actually Represents | What It Does NOT Automatically Represent |
| :--- | :--- | :--- |
| **Fundus Image → DR Severity** | Localized microvascular retinal damage (microaneurysms, hemorrhages, exudates, neovascularization; graded 0–4). | Does **NOT** prove systemic diabetes mellitus onset, fasting plasma glucose level, or HbA1c elevation without systemic lab work. |
| **Renal CT/US → Kidney Stone Finding** | Localized calcification / nephrolithiasis or structural obstruction in the renal pelvis/calyx. | Does **NOT** establish Chronic Kidney Disease (CKD) staging (which requires KDIGO criteria: eGFR < 60 mL/min/1.73m² or persistent albuminuria > 30 mg/g). |
| **12-Lead ECG → Diagnostic Finding** | Electrophysiological conduction delay, rhythm disturbance, repolarization anomaly, or localized myocardial ischemia/infarction pattern. | Does **NOT** establish comprehensive multi-vessel coronary artery disease, structural valve pathology, or heart failure without echocardiography/angiography. |
| **NHANES Lab Panel → Diabetes Label** | Laboratory-confirmed HbA1c ≥ 6.5% or Fasting Glucose ≥ 126 mg/dL or physician diagnosis. | Does **NOT** provide ophthalmic fundus imaging or assess diabetic retinopathy grade. |
| **NHANES Lab Panel → CKD Label** | Estimated Glomerular Filtration Rate (eGFR < 60) calculated via CKD-EPI formula from serum creatinine and BUN. | Does **NOT** provide renal ultrasound or CT radiology scans. |

---

## Step 3 — Detailed Audit of Clinical Datasets

| Dataset File | File Path | Total Records | Patient / Subject / Record IDs | Clinical Features Present | Target Disease Labels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NHANES Diabetes Cleaned** | `NHANES_diabetes_cleaned.csv` | 11,452 | **NONE** (No `patient_id`, no `SEQN`, no `subject_id`, no `record_id`). | `Age`, `Sex`, `BMI`, `Waist`, `Systolic_BP`, `Diastolic_BP`, `HDL`, `Total_Cholesterol`, `Glucose`, `HbA1c` (10 features) | `Diabetes` (0 = No, 1 = Yes) |
| **NHANES Heart Disease Cleaned** | `NHANES_heart_disease_cleaned.csv` | 7,770 | **NONE** (No `patient_id`, no `SEQN`, no `subject_id`, no `record_id`). | `Age`, `Sex`, `BMI`, `Waist`, `Systolic_BP`, `Diastolic_BP`, `HDL`, `Total_Cholesterol`, `Glucose`, `HbA1c` (10 features) | `Heart_Disease` (0 = No, 1 = Yes) |
| **Processed Clinical Partitions** | `data/processed/clinical/train_clinical.csv`<br>`data/processed/clinical/val_clinical.csv`<br>`data/processed/clinical/test_clinical.csv` | 8,016 (Train)<br>1,718 (Val)<br>1,718 (Test)<br>Total: 11,452 | **NONE** (No row identifiers or participant IDs). | `age`, `sex`, `bmi`, `waist`, `systolic_bp`, `diastolic_bp`, `hdl`, `total_cholesterol`, `glucose`, `hba1c`, `creatinine`, `bun` (12 features) | `diabetes`, `heart_disease`, `ckd` (Multi-task binary targets) |
| **PTB-XL Database Metadata** | `data/ptbxl/.../ptbxl_database.csv` | 21,799 | `patient_id` (18,869 unique patients)<br>`ecg_id` (21,799 unique records)<br>`filename_lr`<br>`filename_hr` | `age`, `sex` (0=F, 1=M), `height` (68% missing), `weight` (56.8% missing), `nurse`, `site`, `device`, `heart_axis` | `scp_codes` (71 statements)<br>`target` (0 = Normal, 1 = Abnormal) |

> [!WARNING]
> **No Participant Identifiers in NHANES:**
> In both `NHANES_diabetes_cleaned.csv` and `NHANES_heart_disease_cleaned.csv`, the original CDC NHANES participant sequence number (`SEQN`) was dropped during data cleaning. The rows represent unindexed tabular observation vectors. Row numbers cannot be used as patient identifiers.

---

## Step 4 — Check for Real Linkage (Image ↔ Clinical Record)

A rigorous relational and cryptographic search was conducted across all datasets for shared keys (`patient_id`, `subject_id`, `study_id`, `record_id`, `image_id`, `seqn`).

### Linkage Verification Diagram:

```
[Medical Image Dataset (Retinopathy / Kidney / Chest X-ray)]
                      ↓
           [IMAGE IDENTIFIER]  ───>  (DOES NOT EXIST IN REPOSITORY)
                      ↓
      [PATIENT / SUBJECT IDENTIFIER] ───>  (NO VALID LINKAGE FOUND)
                      ↓
       [NHANES CLINICAL RECORD] ───>  (NO SEQN / NO PATIENT ID)
```

### Result:
**"NO VALID PATIENT-LEVEL LINKAGE FOUND."**

- There are zero medical image datasets in the repository.
- There are zero identifiers connecting NHANES survey records to external images or signals.
- The only genuine linkage in the entire repository is internal to PTB-XL:
  $$\text{PTB-XL: } \text{ecg\_id} \longrightarrow \text{patient\_id} \longrightarrow (\text{age, sex, height, weight}) + \text{12-lead ECG waveform}$$

---

## Step 5 — Check NHANES + Image Data

### Can an image from the current image dataset be matched to a specific NHANES patient?

### **NO.**

### Detailed Scientific & Technical Explanation:
1. **Absence of Image Modalities:** The repository contains no medical image files.
2. **Absence of Participant Keys:** The NHANES CSV files in this repository have stripped the CDC respondent sequence identifier (`SEQN`). Even if public NHANES ophthalmic fundus photos (from NHANES 2005–2008) existed, they could not be joined to these specific CSV files without re-extracting the raw CDC XPT files with intact `SEQN` keys.
3. **Cohort Independence:** CDC NHANES is a continuous cross-sectional epidemiological health survey conducted in the United States. External medical imaging datasets (such as EyePACS, APTOS 2019, Messidor, or Kaggle Kidney Stone datasets) originate from completely independent clinical cohorts in India, France, or hospital radiology PACS systems. Under HIPAA and GDPR de-identification standards, these cohorts have zero intersection and cannot legitimately be linked.

---

## Step 6 — Check PTB-XL (Electrophysiological Waveform Dataset)

| Component | Status in PTB-XL | Description |
| :--- | :--- | :--- |
| **Patient Identifiers** | **PRESENT** | `patient_id` column with 18,869 distinct human patients. |
| **Record Identifiers** | **PRESENT** | `ecg_id` column (1 to 21,837) uniquely identifying each 10-second recording session. |
| **Clinical / Demographics** | **PRESENT** | Patient `age` (range 2–95), `sex` (binary), `height`, `weight`, `device`, `recording_date`. |
| **Biosignal Recordings** | **PRESENT** | 21,799 paired 12-lead ECG waveform files stored in WFDB `.dat` and `.hea` format (100 Hz and 500 Hz). |
| **Diagnostic Labels** | **PRESENT** | Clinician-annotated SCP-ECG statements (`NORM`, `MI`, `STTC`, `CD`, `HYP`). |
| **Genuine Same-Patient Data?** | **YES** | Every ECG record is authentically collected from the documented `patient_id` alongside their demographics. |
| **Is it a Medical Image Dataset?** | **NO** | PTB-XL is an **electrophysiological 1D time-series biosignal dataset**, not a 2D pixel image dataset. |

---

## Step 7 — Repository Dataset Comparison Table

| Dataset | Image / Signal Type | Clinical Data | Patient ID | Image / Record ID | Same-Patient Linkage | Disease Label | Suitable for Multimodal Fusion? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NHANES Diabetes** (`NHANES_diabetes_cleaned.csv`) | None | Demographics + Vitals + Labs (10 cols) | None | None | **NO** | Diabetes (Binary) | **NO** (Clinical Tabular Only) |
| **NHANES Heart Disease** (`NHANES_heart_disease_cleaned.csv`) | None | Demographics + Vitals + Labs (10 cols) | None | None | **NO** | Heart Disease (Binary) | **NO** (Clinical Tabular Only) |
| **Processed Multi-Task Clinical** (`data/processed/clinical/*.csv`) | None | Demographics + Vitals + Labs + Kidney Biomarkers (12 cols) | None | None | **NO** | Diabetes, Heart Disease, CKD | **NO** (Clinical Tabular Only) |
| **PTB-XL Dataset** (`data/ptbxl/`) | 1D Biosignal: 12-Lead ECG Waveforms (100 Hz / 500 Hz) | Demographics: Age, Sex, Height, Weight (4 cols) | `patient_id` (18,869 IDs) | `ecg_id` (21,799 IDs) | **YES** (Internal Demographic + ECG) | ECG Arrhythmia & Diagnostic Abnormality | **YES** (Clinical + 1D Biosignal Waveform Fusion) |
| **Diabetic Retinopathy Images** | None Present | None Present | None | None | **NO** | N/A | **NO** (Dataset Missing) |
| **Kidney / Renal Images** | None Present | None Present | None | None | **NO** | N/A | **NO** (Dataset Missing) |
| **Generic Medical Images** | None Present | None Present | None | None | **NO** | N/A | **NO** (Dataset Missing) |

---

## Step 8 — Scientific Classification of Datasets

Each dataset in the repository is classified into exactly one scientific category:

1. **NHANES Diabetes Dataset (`NHANES_diabetes_cleaned.csv`):**  
   `3. CLINICAL DATA ONLY` (Used in standalone multi-disease clinical pipeline)
2. **NHANES Heart Disease Dataset (`NHANES_heart_disease_cleaned.csv`):**  
   `3. CLINICAL DATA ONLY` (Used in standalone multi-disease clinical pipeline)
3. **Processed Multi-Task Clinical Dataset (`data/processed/clinical/`):**  
   `3. CLINICAL DATA ONLY` (Used to train `models/multidisease_gnn_best.pt`)
4. **PTB-XL ECG Dataset (`data/ptbxl/`):**  
   `1. VALID PATIENT-LEVEL MULTIMODAL DATA` (For Clinical Demographics + 1D ECG Waveform)  
   *and* `6. ALREADY USED IN A SEPARATE MODEL` (Trained and verified in `models/ptbxl_multimodal/best_model.pt`)
5. **Medical Image Datasets (Retinopathy / Kidney / Chest X-ray):**  
   `4. DATA EXISTS BUT NO PATIENT LINKAGE` $\rightarrow$ Specifically: **DATA NOT PRESENT IN REPOSITORY**.

---

## Step 9 — Existing Codebase Implementation vs. Usage Audit

An inspection of all Python preprocessing modules, neural network architectures, graph builders, explainability modules, backend API routes, and React frontend components was conducted:

| Architectural Component | File Location | Implementation Status | Operational Usage in Repo |
| :--- | :--- | :--- | :--- |
| **Image Preprocessing** | `app/preprocessing/image_preprocessor.py` | **IMPLEMENTED BUT NOT USED** | Fully implemented with PIL validation and Torchvision transforms; tested with dummy synthetic image in `test_image_preprocessor.py`; never called during training or inference. |
| **Image Encoder** | `app/models/image_encoder.py` | **IMPLEMENTED BUT NOT USED** | Implemented ResNet18/50 and EfficientNet-B0 backbones with projection head; unit tested with random tensor in `test_encoders.py`; never trained or executed in production. |
| **Clinical Encoder** | `app/models/clinical_encoder.py` | **IMPLEMENTED AND USED** | 2-layer MLP (12 $\rightarrow$ 64 dim for Multi-Disease; 4 $\rightarrow$ 64 dim for PTB-XL) used in `train_multidisease_gnn.py`, `train_ptbxl_multimodal.py`, and runtime inference. |
| **ECG Waveform Encoder** | `app/models/ecg_encoder.py` | **IMPLEMENTED AND USED** | 1D CNN with residual blocks (12 leads $\times$ 1000 samples $\rightarrow$ 128 dim embedding) trained on PTB-XL in `train_ptbxl_multimodal.py`. |
| **Multimodal Fusion** | `app/models/multimodal_model.py` | **IMPLEMENTED AND USED** | Concatenates and projects multimodal representations (used for Clinical + ECG in `train_ptbxl_multimodal.py`; image branch exists but is bypassed via `is_linked_patient=False`). |
| **Patient Graph Construction** | `app/graph/patient_graph.py`<br>`graph/build_graph.py` | **IMPLEMENTED AND USED** | $k$-NN Cosine Similarity graph ($k=5$) constructed during batch training in `train_multidisease_gnn.py` and `train_ptbxl_multimodal.py`. |
| **Graph Neural Network (GNN)** | `app/models/gnn_model.py` | **IMPLEMENTED AND USED** | `MultiDiseaseGNN` with message passing and multi-task prediction heads used in both trained checkpoints. |
| **Image XAI (Grad-CAM)** | `app/explainability/image_xai.py` | **IMPLEMENTED BUT NOT USED** | Implements forward/backward hook Grad-CAM for CNNs; not invoked by backend routes or inference services. |
| **Tabular & ECG XAI** | `explainability/ptbxl_explainer.py`<br>`app/explainability/tabular_xai.py` | **IMPLEMENTED AND USED** | Captum Integrated Gradients for clinical feature attributions and 12-lead ECG waveform attributions. |
| **FastAPI Backend** | `backend/app/main.py`<br>`backend/app/api/routes.py` | **IMPLEMENTED AND USED** | Serves `/predict`, `/metadata`, `/health`, `/evaluate` for multi-disease risk assessment. |
| **React Frontend** | `frontend/src/pages/PatientAssessment.jsx` | **IMPLEMENTED AND USED** | 14-field clinical intake UI, multi-disease risk gauges, XAI feature attribution charts, and ECG radar metrics. |

---

## Step 10 — Final Audit Answers

### 1. What image datasets do I currently have?
**Zero.** There are no 2D medical image datasets (DICOM, NIfTI, PNG, JPG, TIFF) in the repository. The only `.png` files are UI graphics and EDA/evaluation charts. The only biosignal dataset is the PTB-XL 12-lead ECG time-series dataset.

### 2. What clinical datasets do I currently have?
You have two clinical data sources:
1. **NHANES Tabular Data:** `NHANES_diabetes_cleaned.csv` (11,452 rows) and `NHANES_heart_disease_cleaned.csv` (7,770 rows), processed into multi-task clinical partitions in `data/processed/clinical/` (11,452 total rows with 12 features for Diabetes, Heart Disease, and CKD).
2. **PTB-XL Clinical Metadata:** `data/ptbxl/.../ptbxl_database.csv` (21,799 records containing patient `age`, `sex`, `height`, and `weight`).

### 3. What patient identifiers exist?
- **In NHANES:** **Zero patient identifiers exist.** All survey sequence numbers (`SEQN`) and participant IDs were stripped during preprocessing.
- **In PTB-XL:** `patient_id` (18,869 unique IDs) and `ecg_id` (21,799 unique record IDs) exist and link each demographic row to its corresponding 12-lead ECG waveform file.

### 4. Do any image datasets have patient IDs?
**No image datasets exist in the repository.** Therefore, no image datasets have patient IDs.

### 5. Can any image dataset be linked to NHANES patients?
**NO.** No image dataset exists, and the NHANES CSVs contain no participant identifiers. Furthermore, NHANES survey participants are an independent US population cohort with no intersection with external medical imaging datasets.

### 6. Can any image dataset be linked to another clinical dataset?
**NO.** No image datasets exist in the repository.

### 7. Which datasets genuinely contain same-patient clinical + image information?
**None.** There is zero same-patient clinical + 2D medical image data.  
*(Note: PTB-XL genuinely contains same-patient clinical demographics + **1D ECG waveform biosignals**, but not 2D images).*

### 8. Which datasets cannot be paired?
- `NHANES_diabetes_cleaned.csv` cannot be paired with any image or ECG dataset.
- `NHANES_heart_disease_cleaned.csv` cannot be paired with any image or ECG dataset.
- `data/processed/clinical/*.csv` cannot be paired with any image or ECG dataset.
- PTB-XL cannot be paired with NHANES data.

### 9. What disease does each image dataset actually predict?
Since no image dataset is present, no image dataset predicts any disease in the current repo.
- **MultiDiseaseGNN (`models/multidisease_gnn_best.pt`):** Predicts **Diabetes** (AUC 0.947), **Heart Disease** (AUC 0.739), and **CKD** (AUC 0.990) purely from tabular clinical laboratory markers.
- **PTBXLMultimodalGNN (`models/ptbxl_multimodal/best_model.pt`):** Predicts **ECG Arrhythmia & Diagnostic Abnormality** (AUC 0.825) from clinical demographics (`age`, `sex`, `height`, `weight`) + 12-lead ECG time-series waveforms.

### 10. Which dataset is suitable for building:
$$\text{Clinical Data} + \text{Medical Image} \longrightarrow \text{Multimodal Patient Representation} \longrightarrow \text{Patient Similarity Graph} \longrightarrow \text{GNN} \longrightarrow \text{Disease Prediction} \longrightarrow \text{XAI}$$

**None of the current datasets are suitable for Clinical + 2D Medical Image Multimodal Fusion**, because no paired clinical-and-image dataset exists in the repository.

*(PTB-XL is suitable and currently used for **Clinical + 1D ECG Waveform** Multimodal Graph Fusion, but not for 2D images).*

---

### 11. Final Requirement: Required Dataset Specifications

> [!IMPORTANT]
> **"An additional paired clinical + medical-image dataset is required."**

To legitimately train and deploy a **Clinical Data + Medical Image Multimodal Graph Neural Network**, an additional dataset must be introduced that satisfies all of the following strict criteria:

#### Mandatory Dataset Requirements:
1. **Verifiable Same-Patient Linkage:**
   - Every image file must possess a definitive `patient_id` or `study_id` that maps 1-to-1 to a tabular clinical record row for the **exact same human subject**.
   - Example schema: `patient_id` $\rightarrow$ `image_path` (e.g., `fundus_00124.jpg`) $\leftrightarrow$ `clinical_table.csv` (`patient_id`, `age`, `sex`, `hba1c`, `glucose`, `systolic_bp`, `target`).
2. **Clinical Modality Alignment:**
   - **For Diabetic Retinopathy Multimodal Pipeline:** Paired digital color fundus photographs (macula/optic-disc centered, e.g., EyePACS, APTOS 2019, or Messidor-2) linked to genuine patient clinical metrics (HbA1c, fasting glucose, diabetes duration, systolic blood pressure).
   - **For Renal / CKD Multimodal Pipeline:** Paired renal B-mode ultrasound or CT kidney scans linked to genuine patient renal labs (serum creatinine, BUN, eGFR, urine albumin-to-creatinine ratio).
   - **For Pulmonary / Chest Multimodal Pipeline:** Paired Chest X-rays (CXR, e.g., MIMIC-CXR or CheXpert) linked to ICU/ED clinical vitals and demographics (age, sex, SpO2, heart rate, respiratory rate, diagnosis).
3. **No Synthetic or Row-Index Pairing:**
   - The clinical features and images must be collected from the same individual during related medical encounters. No synthetic random matching between unlinked NHANES rows and unlinked Kaggle images can be permitted.

---

```
========================================================================================
AUDIT CONCLUSION:
- Real 2D Medical Image Datasets Present: 0
- Genuine Same-Patient 2D Image + Clinical Linkage: NONE
- Real 1D ECG Biosignal + Clinical Linkage Present: PTB-XL (21,799 records, 18,869 patients)
- Action Taken: Audit report generated without code modification, model retraining, or synthetic linkage.
========================================================================================
```
