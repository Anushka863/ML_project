# Model Feature Mapping & Clinical Variables Audit

**Date:** 2026-09-20  
**Project:** Explainable Multi-Disease Clinical Decision Support System Using Graph Neural Networks  
**Scope:** Academic Traceability of Features Across UI, API, Preprocessing, and Neural Models

---

## 1. Executive Summary

This document audits the exact mapping between variables collected in the clinical intake form, transmitted via the FastAPI REST payload, and ingested by machine learning models.

> [!IMPORTANT]
> **Scientific Integrity Notice:**  
> The web application currently collects a 14-variable cardiometabolic panel designed for chronic disease risk (NHANES). However, the active multimodal GNN model is trained on the PTB-XL electrocardiography dataset, which only includes demographic variables (`age`, `sex`, `height`, `weight`) alongside 12-lead ECG waveforms. Variables not present in PTB-XL are **not** ingested by the neural network weights and are only referenced for rule-based clinical guidance.

---

## 2. Comprehensive Variable Mapping Matrix

| Variable Name | Form Field ID | API Schema Field | PTB-XL Model Input | NHANES Dataset Column | Role in System |
|---|---|---|:---:|:---:|---|
| **Age** | `age` | `age: float` | ✅ **Yes** (Index 0) | `Age` | Demographic feature; standardized and fed into Clinical MLP. |
| **Sex / Gender** | `gender` | `gender: str` | ✅ **Yes** (Index 1) | `Sex` | Encoded as binary float (Male=1.0, Female=0.0, Other=0.48 baseline). |
| **Height** | `height` | `height: Optional[float]` | ✅ **Yes** (Index 2) | — | Standardized; median imputed (166.2 cm baseline) if absent. |
| **Weight** | `weight` | `weight: Optional[float]` | ✅ **Yes** (Index 3) | — | Standardized; median imputed (70.3 kg baseline) if absent. |
| **12-Lead ECG** | *(Signal Input)* | `ecg_waveform` | ✅ **Yes** `(12, 1000)` | — | 10s @ 100Hz 12-lead potential signal ingested by 1D CNN Encoder. In web intake, defaults to zero-signal demo baseline when signal file is omitted. |
| **Body Mass Index** | `bmi` | `bmi: float` | ❌ No | `BMI` | Computed in frontend (`weight / height²`); used in heuristic XAI. |
| **Systolic BP** | `systolic` | `systolic: float` | ❌ No | `Systolic_BP` | Vitals monitoring; used in heuristic rule-based attribution. |
| **Diastolic BP** | `diastolic` | `diastolic: float` | ❌ No | `Diastolic_BP` | Vitals monitoring; used in heuristic rule-based attribution. |
| **Fasting Glucose** | `glucose` | `glucose: float` | ❌ No | `Glucose` | Diabetic metabolic marker; reserved for NHANES diabetes model. |
| **HbA1c** | `hba1c` | `hba1c: float` | ❌ No | `HbA1c` | Glycated hemoglobin; reserved for NHANES diabetes model. |
| **HDL Cholesterol**| `hdl` | `hdl: float` | ❌ No | `HDL` | Lipid profile; reserved for NHANES cardiovascular model. |
| **Total Cholesterol**| `totalCholesterol` | `totalCholesterol: float` | ❌ No | `Total_Cholesterol`| Lipid profile; used in heuristic rule-based attribution. |
| **Serum Creatinine**| `creatinine` | `creatinine: float` | ❌ No | `Creatinine` | Renal function marker; planned for NHANES CKD model. |
| **Blood Urea Nitrogen**| `bun` | `bun: float` | ❌ No | `BUN` | Renal function marker; planned for NHANES CKD model. |
| **Waist Circumference**| `waist` | `waist: Optional[float]`| ❌ No | `Waist` | Anthropometric metric; reserved for NHANES metabolic model. |

---

## 3. Modality & Encoder Alignment

### 3.1. PTB-XL Multimodal GNN (`PTBXLMultimodalGNN`)
- **Clinical Feature Vector ($x_{\text{clinical}} \in \mathbb{R}^4$):**
  $$\mathbf{x}_{\text{clinical}} = [\text{Age}_{\text{norm}}, \text{Sex}_{\text{bin}}, \text{Height}_{\text{norm}}, \text{Weight}_{\text{norm}}]$$
- **ECG Waveform Tensor ($x_{\text{ecg}} \in \mathbb{R}^{12 \times 1000}$):**
  12 standard bipolar and augmented unipolar leads over 10 seconds sampled at 100 Hz.
- **Fusion Layer:**
  $$\mathbf{z}_{\text{patient}} = \text{FusionMLP}([\mathbf{e}_{\text{clinical}} \,\|\, \mathbf{e}_{\text{ecg}}])$$
- **Patient Graph:**
  Cosine similarity $k$-NN graph ($k=5$) over $\mathbf{z}_{\text{patient}}$.

### 3.2. NHANES Tabular Pipeline (Separate Track)
- **Diabetes Dataset (`NHANES_diabetes_cleaned.csv`):** 11,452 rows, target `Diabetes` (9.4% positive).
- **Heart Disease Dataset (`NHANES_heart_disease_cleaned.csv`):** 7,770 rows, target `Heart_Disease` (9.6% positive).
- **CKD Dataset:** Pending acquisition (`NHANES_CKD_cleaned.csv`).
- **Separation Rule:** NHANES patients must never be paired with PTB-XL ECG records.
