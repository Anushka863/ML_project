# MIMIC-CXR + MIMIC-IV Multimodal Branch: Readiness & Access Audit

**Audit Date:** September 22, 2026  
**Auditor:** Antigravity Advanced Agentic AI Assistant  
**Repository:** `ML_PROJECT-Anushka_branch / ML_project`  
**Target Workflow:**
$$\text{Clinical Data} + \text{Chest X-Ray (Same Patient)} \longrightarrow \text{Multimodal Fusion} \longrightarrow \text{Patient Representation} \longrightarrow \text{Patient Similarity Graph} \longrightarrow \text{GNN} \longrightarrow \text{Disease Prediction} \longrightarrow \text{Dual XAI}$$

---

## Executive Audit Summary

This audit evaluates the architectural readiness, module reusability, data governance, storage requirements, and system modifications necessary to integrate an authentic **MIMIC-CXR (Chest Radiographs) + MIMIC-IV (Clinical Vitals & Blood Labs)** multimodal branch into the existing codebase.

---

## 1. Existing Clinical Encoder Reusability

- **File Path:** [`ML_project/app/models/clinical_encoder.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/models/clinical_encoder.py)
- **Component:** `ClinicalEncoder(nn.Module)`
- **Audit Assessment:** **100% REUSABLE AS-IS (ZERO CODE MODIFICATIONS NEEDED)**
- **Technical Details:**
  - Configurable architecture: Takes arbitrary `input_dim`, maps through hidden layers `[128, 64]` with `BatchNorm1d`, `ReLU`, `Dropout(0.2)`, and projects to a fixed `embedding_dim=64`.
  - Supports 1D and 2D tensor inputs with dynamic batching.
  - Can encode any selected MIMIC-IV clinical feature dimension ($D_{clinical} \rightarrow 64$).

---

## 2. Existing Image Encoder Reusability

- **File Path:** [`ML_project/app/models/image_encoder.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/models/image_encoder.py)
- **Component:** `ImageEncoder(nn.Module)`
- **Audit Assessment:** **100% REUSABLE AS-IS (ZERO CODE MODIFICATIONS NEEDED)**
- **Technical Details:**
  - Supports standard torchvision backbones: `resnet18`, `resnet50`, and `efficientnet_b0`.
  - Replaces final classification layer with `nn.Identity()` and routes pooled feature maps into an embedding projection head:
    $$\text{Backbone Features } (D_{in}) \longrightarrow \text{Linear}(D_{in}, 128) \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(0.2) \longrightarrow \text{Linear}(128, 64) \longrightarrow \text{ReLU}$$
  - Fully compatible with standard 3-channel normalized $224 \times 224$ Chest X-ray tensors.

---

## 3. Existing MultimodalFusion Reusability

- **File Path:** [`ML_project/app/models/multimodal_model.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/models/multimodal_model.py)
- **Component:** `MultimodalFusion(nn.Module)`
- **Audit Assessment:** **100% REUSABLE AS-IS (ZERO CODE MODIFICATIONS NEEDED)**
- **Technical Details:**
  - Concatenates clinical embeddings ($d_c = 64$) and image embeddings ($d_i = 64$) into a $128$-dimensional input.
  - Applies a 2-layer Fusion MLP with `BatchNorm1d` and `Dropout`:
    $$\text{Concat}(e_{clinical}, e_{image}) \in \mathbb{R}^{128} \longrightarrow \text{Linear}(128, 256) \longrightarrow \text{BN} \longrightarrow \text{ReLU} \longrightarrow \text{Dropout} \longrightarrow \text{Linear}(256, 128) \longrightarrow \text{BN} \longrightarrow \text{ReLU}$$
  - Features built-in fallback logic via `clinical_projector` when the image modality is missing or unlinked.

---

## 4. Existing PatientGraphBuilder Reusability

- **File Path:** [`ML_project/app/graph/patient_graph.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/graph/patient_graph.py)
- **Component:** `PatientGraphBuilder`
- **Audit Assessment:** **100% REUSABLE AS-IS (ZERO CODE MODIFICATIONS NEEDED)**
- **Technical Details:**
  - Accepts any $N \times D$ matrix of fused patient representations.
  - Constructs $k$-NN graph ($k=5$) using Cosine or Euclidean distance metrics.
  - Outputs PyTorch Geometric `Data(x, edge_index, y)` objects or standalone tensor dictionaries.
  - Includes isolated single-patient graph handling ($N=1, E=0$) for real-time web inference without cross-patient leakage.

---

## 5. Existing GNN Components Reusability

- **File Path:** [`ML_project/app/models/gnn_model.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/models/gnn_model.py)
- **Component:** `MultiDiseaseGNN(nn.Module)` and `SimpleGNNConv(nn.Module)`
- **Audit Assessment:** **100% REUSABLE AS-IS (ZERO CODE MODIFICATIONS NEEDED)**
- **Technical Details:**
  - Multi-layer graph convolutional message passing over graph edges.
  - Multi-head output dictionary `heads = nn.ModuleDict({...})` that dynamically constructs binary logit classifiers for any specified list of target diseases (e.g. `["cardiomegaly", "pulmonary_edema", "pneumonia", "pleural_effusion"]`).

---

## 6. Existing XAI Components Reusability

### 6.1 Clinical Tabular XAI
- **File Path:** [`ML_project/app/explainability/tabular_xai.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/explainability/tabular_xai.py)
- **Component:** `TabularXAI`
- **Audit Assessment:** **100% REUSABLE AS-IS**
- Computes Gradient $\times$ Input / attributions on MIMIC-IV clinical biomarkers and provides directional impact annotations ("increases risk" / "lowers risk").

### 6.2 Radiographic Image XAI
- **File Path:** [`ML_project/app/explainability/image_xai.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/explainability/image_xai.py)
- **Component:** `ImageGradCAM`
- **Audit Assessment:** **100% REUSABLE AS-IS**
- Extracts forward activations and backward pooled gradients from the final convolutional layer of the image backbone (e.g., `backbone.layer4` in ResNet). Generates normalized $[0, 1]$ 2D heatmaps overlaid on the input Chest X-ray.

### 6.3 Graph XAI
- **File Path:** [`ML_project/app/explainability/graph_xai.py`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/app/explainability/graph_xai.py)
- **Component:** `GraphXAI`
- **Audit Assessment:** **100% REUSABLE AS-IS**
- Identifies nearest computational neighbors and edge similarities with honest disclaimer tags.

---

## 7. Required Preprocessing Pipelines

### 7.1 MIMIC-IV Clinical Data Preprocessing
- **Source Tables:** `mimiciv_hosp.patients`, `mimiciv_hosp.admissions`, `mimiciv_hosp.labevents`, `mimiciv_icu.chartevents`.
- **Pipeline Tasks:**
  1. **Temporal Alignment:** Filter laboratory events and charted vitals within $\pm 24\text{ hours}$ (or $\pm 48\text{ hours}$) of the chest radiograph acquisition time (`study_date`, `study_time`).
  2. **Aggregation:** Calculate mean / most recent biomarker values for the clinical window.
  3. **Outlier Filtering & Imputation:** Truncate extreme unphysiological measurement errors and impute missing values using training-set medians.
  4. **Standardization:** Fit `StandardScaler` and `OneHotEncoder` strictly on the training partition.

### 7.2 MIMIC-CXR Image Preprocessing
- **Source Tables:** `mimic-cxr-2.0.0-metadata.csv.gz`, `files/pXX/pXXXXXXXX/sXXXXXXXX/*.jpg`.
- **Pipeline Tasks:**
  1. **View Orientation Filtering:** Select exclusively Frontal projections (`ViewPosition` in `['PA', 'AP']`); discard Lateral views (`'LATERAL'`, `'LL'`) to avoid anatomical distortion.
  2. **Grayscale to RGB Channel Broadcast:** Convert 1-channel grayscale CXR images to 3-channel tensors $(3 \times H \times W)$ for ImageNet-pretrained weights.
  3. **Geometric Transformation:** Resize to $224 \times 224$ pixels, center crop, and apply ImageNet mean/std normalization:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  4. **Data Augmentation (Training Only):** Random horizontal flip ($p=0.5$), slight rotation ($\pm 10^\circ$), and mild brightness jitter.

---

## 8. Clinical Features Selected (Strictly Verified in MIMIC-IV)

The clinical feature set is selected **exclusively** from parameters charted in MIMIC-IV (`patients`, `admissions`, `labevents`, `chartevents`):

| # | Feature Name | Variable Code / Item ID | Unit | Clinical Significance |
|---|---|---|---|---|
| 1 | `age` | `anchor_age` | Years | Baseline cardiovascular risk factor |
| 2 | `gender` | `gender` (M/F) | Categorical | Biological sex stratification |
| 3 | `heart_rate` | `chartevents` (220045) | bpm | Tachycardia / arrhythmia marker |
| 4 | `systolic_bp` | `chartevents` (220179 / 220050) | mmHg | Hypertension / afterload marker |
| 5 | `diastolic_bp` | `chartevents` (220180 / 220051) | mmHg | Coronary perfusion pressure |
| 6 | `respiratory_rate` | `chartevents` (220210) | breaths/min | Tachypnea / respiratory distress |
| 7 | `spo2` | `chartevents` (220277) | % | Hypoxemia indicator |
| 8 | `temperature` | `chartevents` (223761 / 223762) | °C | Systemic inflammation / fever |
| 9 | `glucose` | `labevents` (50931 / 50809) | mg/dL | Glycemic status / stress response |
| 10 | `creatinine` | `labevents` (50912) | mg/dL | Renal function / cardiorenal failure |
| 11 | `bun` | `labevents` (51006) | mg/dL | Blood urea nitrogen / hydration |
| 12 | `sodium` | `labevents` (50983) | mEq/L | Electrolyte balance |
| 13 | `potassium` | `labevents` (50971) | mEq/L | Cardiac excitability marker |
| 14 | `wbc` | `labevents` (51301) | K/uL | Infection / leukocytosis |
| 15 | `hemoglobin` | `labevents` (51222) | g/dL | Oxygen delivery / anemia |
| 16 | `platelets` | `labevents` (51265) | K/uL | Hemostasis marker |
| 17 | `troponin_t` | `labevents` (51003) | ng/mL | Myocardial necrosis biomarker |
| 18 | `lactate` | `labevents` (50813) | mmol/L | Tissue hypoperfusion marker |

*(Note: Outpatient-only features like waist circumference are excluded as they are not charted in acute hospital datasets).*

---

## 9. Available Image Pathology Targets

From `mimic-cxr-2.0.0-chexpert.csv.gz` (14 CheXpert label categories):
1. **Cardiomegaly** (Enlarged Cardiac Silhouette)
2. **Edema** (Pulmonary Edema / Fluid Overload)
3. **Pneumonia** (Pulmonary Infection)
4. **Atelectasis** (Lung Collapse)
5. **Pleural Effusion** (Fluid in Pleural Cavity)
6. **Consolidation** (Alveolar Airspace Opacification)
7. **Pneumothorax** (Pleural Air Leak)
8. **Lung Lesion** (Nodule / Mass)
9. **Lung Opacity** (Non-specific Infiltrate)
10. **Enlarged Cardiomediastinum**
11. **Fracture** (Rib / Clavicle Fracture)
12. **Pleural Other** (Thickening / Plaques)
13. **Support Devices** (PICC Lines, Endotracheal Tubes, Pacemakers)
14. **No Finding** (Normal Radiograph)

---

## 10. Recommended Target for Proof-of-Concept Multimodal Model

### Primary Target: **Cardiomegaly (Enlarged Heart)** or **Congestive Heart Failure / Edema**
- **Biological Justification:** True multi-modal synergy exists between clinical signs (hypertension, elevated troponin, elevated creatinine/BUN, tachycardia) and radiographic manifestations (cardiothoracic ratio $> 0.5$ on Frontal CXR).
- **Dataset Balance:** Cardiomegaly is present in ~27% of MIMIC-CXR frontal studies, providing strong class balance without severe sparsity.
- **Explainability Validation:** Grad-CAM visually highlights the enlarged cardiac margins on the CXR, while Integrated Gradients highlights the corresponding blood pressure and cardiac biomarkers.

---

## 11. Exact Patient & Study Linking Identifiers

To link MIMIC-IV clinical records with MIMIC-CXR images:
1. `subject_id` (Integer, e.g. `10000032`): Global patient identifier spanning both MIMIC-IV and MIMIC-CXR.
2. `study_id` (Integer, e.g. `50414267`): Unique identifier for the chest radiography examination session.
3. `dicom_id` (String UUID, e.g. `02aa804e-bde0afdd-112c3bc3-45541e4c-97e86978`): Unique identifier for the individual image file.
4. `study_date` + `study_time`: Radiograph acquisition timestamp used to match temporal `charttime` in `labevents` and `chartevents`.

$$\text{MIMIC-CXR Metadata } (\text{dicom\_id}, \text{study\_id}, \text{subject\_id}) \xleftrightarrow{\text{subject\_id} + \Delta t} \text{MIMIC-IV Clinical } (\text{subject\_id}, \text{charttime})$$

---

## 12. Train / Validation / Test Splitting Strategy

- **Strategy:** **Patient-Level Group Stratified Splitting** (`GroupShuffleSplit` on `subject_id`) or using the official `mimic-cxr-2.0.0-split.csv.gz`.
- **Guarantee:** Zero patient leakage:
  $$\text{Train}(\text{subject\_id}) \cap \text{Val}(\text{subject\_id}) = \emptyset, \quad \text{Train}(\text{subject\_id}) \cap \text{Test}(\text{subject\_id}) = \emptyset, \quad \text{Val}(\text{subject\_id}) \cap \text{Test}(\text{subject\_id}) = \emptyset$$
- All preprocessing transformations (imputer, scaler, normalizer) are fitted strictly on the training partition.

---

## 13. Data Download Requirements

For a proof-of-concept cohort (e.g. 5,000 paired patients):
1. `mimic-cxr-2.0.0-metadata.csv.gz` (~15 MB)
2. `mimic-cxr-2.0.0-chexpert.csv.gz` (~10 MB)
3. `mimic-cxr-2.0.0-split.csv.gz` (~5 MB)
4. Frontal `.jpg` image files for the selected 5,000 studies from `MIMIC-CXR-JPG`
5. MIMIC-IV core CSVs: `patients.csv.gz`, `admissions.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz` (~1.5 GB compressed)

---

## 14. Access and Credentialing Requirements

- **Platform:** PhysioNet ([https://physionet.org/](https://physionet.org/))
- **Requirements:**
  1. Complete the free CITI training course *"Data or Specimens Only Research"* (~2 hours).
  2. Submit training completion certificate to PhysioNet account.
  3. Electronically sign the PhysioNet Credentialed Data Use Agreement (DUA).
  4. Submit access request for `mimiciv` (v2.2) and `mimic-cxr-jpg` (v2.0.0).
- **Timeline:** Account approval typically takes 24–48 hours after CITI submission.

---

## 15. Storage Footprint Estimates

| Cohort Scope | CXR Image Resolution | Image Storage | Tabular Data | Checkpoints & Graphs | Total Storage |
|---|---|---|---|---|---|
| **PoC Cohort (5,000 Paired Patients)** | $224 \times 224$ (JPEG) | ~250 MB | ~30 MB | ~200 MB | **< 1 GB** |
| **Extended PoC (10,000 Paired Patients)** | $224 \times 224$ (JPEG) | ~500 MB | ~60 MB | ~350 MB | **~1.5 GB** |
| **High-Res PoC (5,000 Paired Patients)** | $512 \times 512$ (JPEG) | ~1.8 GB | ~30 MB | ~200 MB | **~2.2 GB** |
| **Full MIMIC-CXR Dataset (377,110 Scans)** | Full Resolution | ~550 GB | ~25 GB | ~5 GB | **~580 GB** |

*Recommendation:* The 5,000-patient PoC cohort ($224 \times 224$) requires **< 1 GB** total storage, perfectly suited for local execution.

---

## 16. Existing Files Requiring Modification

| File Path | Component | Proposed Modification |
|---|---|---|
| `backend/app/api/schemas.py` | API Schemas | Add CXR upload parameters, `image_xai` output schema, and multimodal prediction models. |
| `backend/app/api/routes.py` | FastAPI Router | Add `POST /predict-multimodal-cxr` (or multipart `POST /predict`) and update `/model-info`. |
| `backend/app/services/prediction_service.py` | Service Layer | Integrate multimodal inference dispatcher linking clinical tabular inputs with CXR image tensors. |
| `frontend/src/pages/PatientAssessment.jsx` | React Intake | Incorporate `ImageUploadSection` for drag-and-drop Chest X-ray file intake with sample selector. |
| `frontend/src/pages/ResultsPage.jsx` | React Results | Add "🫁 Multimodal CXR Analysis" tab with side-by-side Grad-CAM heatmap overlay and clinical attributions. |
| `frontend/src/services/api.js` | API Client | Add multipart `FormData` submission support for image + JSON payload. |

---

## 17. Checkpoints That MUST NOT Be Modified

The following trained model checkpoints and preprocessors **MUST REMAIN 100% UNTOUCHED**:

1. [`models/multidisease_gnn_best.pt`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/multidisease_gnn_best.pt) — Clinical Multi-Disease GNN (Diabetes, Heart Disease, CKD; ROC-AUC: 0.9470, 0.7388, 0.9896).
2. [`models/clinical_preprocessor.joblib`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/clinical_preprocessor.joblib) — Fitted preprocessor for 12-feature NHANES clinical intake.
3. [`models/ptbxl_multimodal/best_model.pt`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/ptbxl_multimodal/best_model.pt) — Independent 12-Lead ECG GNN model.
4. [`models/ptbxl_multimodal/ecg_encoder.pt`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/ptbxl_multimodal/ecg_encoder.pt) — 1D CNN Waveform Encoder.
5. [`models/ptbxl_multimodal/preprocessor.joblib`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/ptbxl_multimodal/preprocessor.joblib) — PTB-XL demographic preprocessor.
6. [`models/ptbxl_multimodal/ptbxl_classifier.pt`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/models/ptbxl_multimodal/ptbxl_classifier.pt) — PTB-XL ECG classifier weights.

---

## 18. New Files / Modules Required

1. `ml_pipeline/prepare_mimic_multimodal.py`: Script to extract, temporally align, and export paired MIMIC-IV clinical labs and MIMIC-CXR frontal image paths.
2. `ml_pipeline/train_mimic_cxr_multimodal.py`: Training script for end-to-end Multimodal GNN training on the paired cohort.
3. `app/inference/mimic_cxr_inference.py`: Inference service linking image tensor and clinical vector through `ClinicalEncoder` + `ImageEncoder` + `MultimodalFusion` + `PatientGraphBuilder` + `MultiDiseaseGNN`.
4. `frontend/src/components/assessment/ImageUploadSection.jsx`: UI component for Chest X-ray image upload with drag-and-drop, image preview, validation, and sample CXR presets.
5. `models/mimic_cxr_multimodal/`: Directory to store newly trained multimodal weights (e.g. `best_model.pt`, `mimic_preprocessor.joblib`).

---

## 19. React Workflow Extension Feasibility

$$\text{Patient Clinical Information} \longrightarrow \text{Upload Chest X-ray} \longrightarrow \text{Analyze} \longrightarrow \text{Multimodal Result}$$

- **Feasibility:** **100% FEASIBLE & SEAMLESS**
- The React application is structured modularly:
  - `PatientAssessment.jsx`: Can host an optional `ImageUploadSection` alongside existing basic info, vitals, and lab panels.
  - `PatientReview.jsx`: Displays uploaded radiograph preview alongside clinical tabular summary.
  - `ResultsPage.jsx`: Displays multimodal prediction scores, clinical XAI (Integrated Gradients bar chart), radiographic XAI (interactive Grad-CAM heatmap overlay with transparency slider), and graph similarity metrics.

---

## 20. Honest Limitations to Display to Users

1. **Decision Support Only:** The system is an investigative research decision support tool, not an FDA-cleared diagnostic device or replacement for a certified radiologist.
2. **Inference Graph State ($N=1, E=0$):** Single-patient web predictions use isolated graph embedding without cross-patient message passing to prevent real-time data contamination.
3. **Population Distribution:** MIMIC-IV / MIMIC-CXR reflects an acute emergency and ICU inpatient population at a major urban US academic medical center (Beth Israel Deaconess Medical Center, Boston).
4. **Acquisition Sensitivity:** Image analysis assumes standard Frontal (PA/AP) chest radiographs; poor exposure, lordotic views, or foreign body artifacts can influence heatmap attributions.
5. **Fallback Behavior:** If no Chest X-ray is uploaded, the system executes in clinical-only mode and explicitly notes the absence of the imaging modality.

---

## Summary Matrix

### A. READY TO IMPLEMENT
- Architectural code components are **100% ready**: `ClinicalEncoder`, `ImageEncoder`, `MultimodalFusion`, `PatientGraphBuilder`, `MultiDiseaseGNN`, `TabularXAI`, `ImageGradCAM`, `GraphXAI`.
- React frontend architecture is ready to accommodate the `ImageUploadSection` and CXR Grad-CAM visualization tab.

### B. BLOCKED BY DATA ACCESS
- Real MIMIC-CXR-JPG and MIMIC-IV data downloads require active **PhysioNet CITI Credentialing** and signed Data Use Agreement (DUA).
- Downloading or training cannot proceed until valid PhysioNet credentials or a pre-downloaded local copy is provided.

### C. EXISTING COMPONENTS REUSABLE
- `app/models/clinical_encoder.py` (`ClinicalEncoder`)
- `app/models/image_encoder.py` (`ImageEncoder`)
- `app/models/multimodal_model.py` (`MultimodalFusion`)
- `app/graph/patient_graph.py` (`PatientGraphBuilder`)
- `app/models/gnn_model.py` (`MultiDiseaseGNN`, `SimpleGNNConv`)
- `app/explainability/tabular_xai.py` (`TabularXAI`)
- `app/explainability/image_xai.py` (`ImageGradCAM`)
- `app/explainability/graph_xai.py` (`GraphXAI`)
- `app/preprocessing/image_preprocessor.py` (`ImagePreprocessor`)

### D. NEW COMPONENTS REQUIRED
- `ml_pipeline/prepare_mimic_multimodal.py` (MIMIC-IV + MIMIC-CXR data joiner)
- `ml_pipeline/train_mimic_cxr_multimodal.py` (Multimodal GNN trainer)
- `app/inference/mimic_cxr_inference.py` (Inference engine)
- `frontend/src/components/assessment/ImageUploadSection.jsx` (React UI component)
- `models/mimic_cxr_multimodal/` (New checkpoint folder)

### E. EXISTING COMPONENTS THAT MUST NOT BE MODIFIED
- `models/multidisease_gnn_best.pt` (Clinical Multi-Disease GNN)
- `models/clinical_preprocessor.joblib` (NHANES clinical preprocessor)
- `models/ptbxl_multimodal/` (All 5 PTB-XL ECG model checkpoints and preprocessors)
- Existing multi-disease inference pipelines (`app/inference/multidisease_inference.py`)
