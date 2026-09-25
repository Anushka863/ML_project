# Paired Multimodal (Clinical + Medical Image) Dataset Research Report

**Date of Report:** September 22, 2026  
**Auditor / Researcher:** Antigravity Advanced Agentic AI Assistant  
**Repository:** `ML_PROJECT-Anushka_branch / ML_project`  
**Target Architecture:**
$$\text{Clinical Data} + \text{Medical Image (Same Patient)} \longrightarrow \text{Clinical/Image Encoders} \longrightarrow \text{Multimodal Fusion} \longrightarrow \text{Patient Similarity Graph} \longrightarrow \text{GNN} \longrightarrow \text{Disease Prediction} \longrightarrow \text{XAI}$$

---

## Executive Summary

Following the repository audit confirming the absence of 2D medical images and the lack of patient identifiers in the preprocessed NHANES files, this research report conducts an authoritative investigation into potential real-world datasets that provide **authentic, same-patient linkage between clinical tabular information and 2D medical images**.

### Summary of Findings by Disease Domain:

1. **Diabetes / Diabetic Retinopathy:**
   - Popular benchmark datasets (**APTOS 2019**, **EyePACS**, **Messidor-2**, **DDR**, **IDRiD**) are **Image-Only datasets** (Category B). They provide only fundus images paired with an ophthalmic DR severity grade (0–4); they do **not** contain patient-level systemic clinical parameters (no HbA1c, no fasting blood glucose, no blood pressure, no lipid panel).
   - **ODIR-5K** is a **Lightweight Paired Multimodal Dataset** (Category A / Demographics): Provides 10,000 fundus images (left and right eye) across 5,000 distinct patients with patient `ID`, `Patient Age`, `Patient Sex`, and multi-label systemic/ocular disease classifications (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other).

2. **Kidney / Renal Disease:**
   - Popular datasets (**CT Kidney Dataset**, **Kaggle Kidney Stone Ultrasound**, **KiTS19/21/23**) are **Image-Only or Oncology Segmentation datasets** (Category B).
   - None of the publicly available renal imaging datasets provide paired systemic laboratory panels (serum creatinine, BUN, eGFR, urine albumin) for Chronic Kidney Disease (CKD) staging linked to the same human patient.

3. **Heart / Cardiac Disease:**
   - **EchoNet-Dynamic (Stanford):** A **Genuine Paired Multimodal Dataset** (Category A). Contains 10,030 apical-4-chamber echocardiography imaging studies from 10,030 individual patients linked to patient `Age`, `Sex`, Left Ventricular Ejection Fraction (`EF`), End-Diastolic Volume (`EDV`), and End-Systolic Volume (`ESV`). Available via standard research DUA.
   - **UK Biobank CMR:** Highest quality paired dataset, but access is **heavily restricted / fee-gated** (Category E for rapid academic prototyping).

4. **Chest / Pulmonary Disease:**
   - **MIMIC-CXR linked with MIMIC-IV (MIT / Beth Israel Deaconess / PhysioNet):** The **Gold Standard Paired Multimodal Healthcare Dataset** (Category A). Contains 377,110 chest radiographs across 65,379 unique patients, linked directly via `subject_id` and `study_id` to complete MIMIC-IV clinical databases (demographics, vital signs, comprehensive blood chemistry/lab events, medications, and ICD diagnoses). Fully open for credentialed academic research (PhysioNet CITI training).
   - **COVID-19-NY-SBU (Stony Brook / TCIA):** A **Genuine Paired Multimodal Dataset** (Category A). Contains 1,384 patients with 5,000+ Chest X-rays linked via `PatientID` to extensive clinical demographics, vitals, blood labs, and COVID-19 severity/mortality outcomes.

---

## Dataset Classification Categories

Every candidate dataset is evaluated and assigned to exactly one of the following scientific categories:

- **Category A: Genuine Paired Multimodal Dataset** (Verifiable same-patient key linking clinical tabular features and medical images).
- **Category B: Image-Only Dataset** (Images + diagnostic classification or bounding box only; no systemic patient clinical variables).
- **Category C: Clinical-Only Dataset** (Tabular clinical/lab data only; no medical images).
- **Category D: Dataset Where Linkage is Unclear** (Ambiguous identifiers, missing crosswalk tables, or unverified pairing).
- **Category E: Dataset Where Linkage is Not Allowed / Restricted** (Linkage exists in proprietary hospital databases but is legally prohibited or inaccessible for academic projects).

---

## Deep-Dive Analysis of Candidate Datasets

---

### Domain 1: Diabetes & Diabetic Retinopathy (DR)

#### 1.1 APTOS 2019 Blindness Detection
- **Official Source:** Kaggle / Aravind Eye Hospital, India ([Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection))
- **Image Type:** Digital retinal fundus photography (`.png`)
- **Dataset Size:** 3,662 training images; 1,928 test images
- **Patient Identifiers:** `id_code` (e.g., `000c1434d8d7`) representing the image file name. No patient ID provided (multiple images per patient cannot be distinguished).
- **Clinical Variables Available:** **NONE.** No age, sex, duration of diabetes, HbA1c, blood glucose, or blood pressure.
- **Disease / Target Labels:** `diagnosis` integer from 0 to 4 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
- **Clinical Distinction:** Measures localized microvascular retinal lesion grade, **not** systemic glycemic control or diabetes diagnosis.
- **Scientific Classification:** **Category B: Image-Only Dataset**
- **Suitability for Multimodal GNN:** **UNSUITABLE** (No clinical modality exists).

#### 1.2 EyePACS (Kaggle Diabetic Retinopathy Detection)
- **Official Source:** EyePACS / California Telehealth Network ([Kaggle EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection))
- **Image Type:** High-resolution digital fundus images (`.jpeg`)
- **Dataset Size:** 88,702 images (35,126 train, 53,576 test)
- **Patient Identifiers:** Images named by `[patient_id]_[left/right].jpeg` (e.g., `10_left.jpeg`, `10_right.jpeg`). Patient ID is present in filename, but no clinical record is provided.
- **Clinical Variables Available:** **NONE.**
- **Disease / Target Labels:** `level` (0 to 4 DR severity).
- **Scientific Classification:** **Category B: Image-Only Dataset**
- **Suitability for Multimodal GNN:** **UNSUITABLE** (No clinical tabular features).

#### 1.3 Messidor-2
- **Official Source:** Messidor Consortium / LaTIM ([Messidor-2](https://www.adcis.net/en/third-party/messidor2/))
- **Image Type:** Color fundus photography
- **Dataset Size:** 1,748 images from 874 examinations
- **Patient Identifiers:** `image_id` and examination identifiers.
- **Clinical Variables Available:** **NONE** in standard release (Only eye side, DR grade, and Diabetic Macular Edema risk).
- **Scientific Classification:** **Category B: Image-Only Dataset**
- **Suitability for Multimodal GNN:** **UNSUITABLE** (Image only).

#### 1.4 ODIR-5K (Ocular Disease Intelligent Recognition)
- **Official Source:** Shanggong Medical Technology / Peking University ([Grand Challenge ODIR-2019](https://odir2019.grand-challenge.org/))
- **Image Type:** Color fundus photographs (Left eye + Right eye per patient)
- **Dataset Size:** 10,000 images from **5,000 unique human patients**
- **Patient Identifiers:** `ID` (Integer patient index 0 to 4999), `Left-Fundus` filename, `Right-Fundus` filename.
- **Clinical Variables Available:**
  - `Patient Age` (Integer)
  - `Patient Sex` (Male / Female)
  - `Left-Diagnostic Keywords` (Doctor clinical text notes)
  - `Right-Diagnostic Keywords` (Doctor clinical text notes)
- **Disease / Target Labels:** 8 multi-label disease flags at patient level:
  - `N`: Normal
  - `D`: Diabetes (Diabetic Retinopathy)
  - `G`: Glaucoma
  - `C`: Cataract
  - `A`: Age-related Macular Degeneration (AMD)
  - `H`: Hypertension (Hypertensive Retinopathy)
  - `M`: Pathological Myopia
  - `O`: Other diseases
- **Patient-Level Linkage Evidence:**
  $$\text{Patient ID} \longrightarrow (\text{Patient Age}, \text{Patient Sex}) + \text{Left-Fundus Image} + \text{Right-Fundus Image} \longrightarrow \text{Multi-Label Disease Vector}$$
- **Public Accessibility & License:** Publicly downloadable on Kaggle and Grand-Challenge for non-commercial academic research.
- **Academic Feasibility:** **HIGH** (Ready-to-use CSV + JPEG directory, ~1.5 GB).
- **Patient-Level Splitting:** **YES** (Split by unique patient `ID` ensures zero leakage).
- **Scientific Classification:** **Category A: Genuine Paired Multimodal Dataset (Demographic + Bilateral Imaging)**
- **Suitability for Multimodal GNN:** **SUITABLE (Lightweight Multimodal)**. Supports Clinical Encoder (Age, Sex, text embeddings) + Image Encoder (ResNet/EfficientNet for bilateral fundus) $\rightarrow$ Multimodal Fusion $\rightarrow$ Patient Graph $\rightarrow$ Multi-Label Disease GNN.
- **Limitations:** Clinical features are limited to demographics (Age, Sex) and clinical keyword text; does not include serum lab panels (no HbA1c, glucose, or lipid chemistry).

---

### Domain 2: Kidney & Renal Disease

#### 2.1 CT KIDNEY DATASET (Kaggle)
- **Official Source:** Kaggle ([CT Kidney Dataset Normal-Cyst-Tumor-Stone](https://www.kaggle.com/datasets/nazmulhasansabbir/ct-kidney-dataset-normal-cyst-tumor-and-stone))
- **Image Type:** Axial CT slices (`.jpg` / `.png`)
- **Dataset Size:** 12,446 CT slice images
- **Patient Identifiers:** **NONE.** Folders organized purely by disease category (`Cyst`, `Normal`, `Stone`, `Tumor`). Filenames are arbitrary slice numbers.
- **Clinical Variables Available:** **NONE.** No serum creatinine, no BUN, no eGFR, no age, no sex.
- **Scientific Classification:** **Category B: Image-Only Dataset**
- **Suitability for Multimodal GNN:** **UNSUITABLE** (No patient IDs, no clinical variables).

#### 2.2 Kaggle Kidney Stone Ultrasound
- **Official Source:** Kaggle / Various Hospitals
- **Image Type:** B-Mode Renal Ultrasound images
- **Dataset Size:** ~1,500 ultrasound frames
- **Patient Identifiers:** **NONE.**
- **Clinical Variables Available:** **NONE.**
- **Scientific Classification:** **Category B: Image-Only Dataset**
- **Suitability for Multimodal GNN:** **UNSUITABLE**.

#### 2.3 KiTS19 / KiTS21 / KiTS23 (Kidney Tumor Segmentation Challenge)
- **Official Source:** University of Minnesota / MICCAI ([KiTS Challenge](https://kits-challenge.org/))
- **Image Type:** 3D Abdominal CT Volumes (`.nii.gz`)
- **Dataset Size:** 300–500 patients
- **Patient Identifiers:** `case_00000` to `case_00499`
- **Clinical Variables Available:** Surgical metadata (Age, Sex, BMI, surgical approach, tumor histology).
- **Disease / Target Labels:** Renal cell carcinoma staging and tumor/cyst/kidney segmentation masks.
- **Scientific Classification:** **Category B: Oncology Imaging Benchmark with Metadata**
- **Suitability for Multimodal GNN:** **POOR FOR GENERAL RENAL/CKD PREDICTION** (Oncology-focused; 3D NIfTI CT volumes require heavy GPU compute; lacks standard CKD biomarkers like serial creatinine and BUN).

---

### Domain 3: Heart & Cardiac Imaging

#### 3.1 EchoNet-Dynamic (Stanford University)
- **Official Source:** Stanford Machine Learning Group / Stanford University School of Medicine ([EchoNet-Dynamic](https://echonet.github.io/dynamic/))
- **Official Citation:** Ouyang, D., et al. "Video-based AI for beat-to-beat cardiac function assessment." *Nature* 580, 252–256 (2020).
- **Image Type:** Apical-4-Chamber (A4C) Echocardiography Videos / Keyframe Images (`.avi` / `.mp4` / extracted `.png` frames)
- **Dataset Size:** **10,030 echocardiogram studies from 10,030 distinct patients**
- **Patient Identifiers:** `FileName` (maps 1-to-1 to each unique patient study).
- **Clinical Variables Available:**
  - `Age` (Integer years)
  - `Sex` (Male / Female)
  - `EF` (Left Ventricular Ejection Fraction, continuous percentage, e.g., 55.4%)
  - `EDV` (End-Diastolic Volume in mL)
  - `ESV` (End-Systolic Volume in mL)
- **Disease / Target Labels:**
  - Heart Failure / Left Ventricular Systolic Dysfunction ($\text{EF} < 50\%$ or continuous EF regression)
  - Cardiac Chamber Enlargement (EDV/ESV thresholds)
- **Patient-Level Linkage Evidence:**
  $$\text{FileName / Patient} \longrightarrow (\text{Age}, \text{Sex}, \text{EDV}, \text{ESV}) + \text{Echocardiogram Frame/Video} \longrightarrow \text{Ejection Fraction / Heart Failure Label}$$
- **Public Accessibility & License:** Freely available for non-commercial academic research via the Stanford Research Data Use Agreement (instant click-through DUA).
- **Academic Feasibility:** **VERY HIGH** (~35 GB for videos, or ~3 GB for extracted end-systolic/end-diastolic 2D keyframes).
- **Patient-Level Splitting:** **YES** (Standard train/val/test splits provided with zero patient overlap).
- **Scientific Classification:** **Category A: Genuine Paired Multimodal Dataset (Clinical Vitals/Volumes + Cardiac Imaging)**
- **Suitability for Multimodal GNN:** **EXCELLENT**. Clinical features ($\text{Age}, \text{Sex}, \text{EDV}, \text{ESV}$) encoded via MLP; 2D end-diastolic/systolic echo frame encoded via ResNet18/EfficientNet; fused representation builds patient graph to predict Heart Failure / Systolic Dysfunction.
- **Limitations:** Focuses specifically on echocardiographic chamber mechanics; does not include lipid blood chemistry.

#### 3.2 UK Biobank Cardiac Magnetic Resonance (CMR)
- **Official Source:** UK Biobank ([UK Biobank Imaging](https://www.ukbiobank.ac.uk/))
- **Image Type:** Multi-view Cardiac MRI (Short-axis, long-axis 2D/3D DICOM)
- **Dataset Size:** 40,000+ scanned participants
- **Patient Identifiers:** Anonymous `eid` linking CMR scans to complete NHS clinical records.
- **Clinical Variables Available:** Exhaustive (complete blood chemistry, genetics, vitals, medical history, lifestyle).
- **Disease / Target Labels:** Myocardial infarction, cardiomyopathy, heart failure, arrhythmia.
- **Public Accessibility & License:** **RESTRICTED.** Requires formal institutional research application, project approval by scientific committee, and payment of data access fee (£3,000–£9,000). 3–6 month turnaround time.
- **Scientific Classification:** **Category E: Genuine Multimodal but Access-Restricted for Rapid Prototyping**
- **Suitability for Multimodal GNN:** Exceptional quality, but impractical for immediate project execution.

---

### Domain 4: Chest & Pulmonary Disease

#### 4.1 MIMIC-CXR-JPG Paired with MIMIC-IV (MIT / BIDMC / PhysioNet)
- **Official Source:** PhysioNet / MIT Laboratory for Computational Physiology ([MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) & [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/))
- **Official Citation:** Johnson, A.E.W., et al. "MIMIC-CXR-JPG - chest radiographs with structured labels." *PhysioNet* (2019).
- **Image Type:** Frontal (PA/AP) and Lateral Chest Radiographs (2D `.jpg` format, 224×224 / 512×512 resolution versions available)
- **Dataset Size:** **377,110 chest X-rays across 65,379 unique human patients (227,835 imaging study sessions)**
- **Patient Identifiers:**
  - `subject_id`: Unique human patient identifier across all MIMIC modules.
  - `study_id`: Unique radiology study session identifier.
  - `dicom_id`: Unique image file identifier.
- **Clinical Variables Available (via relational join to MIMIC-IV `hosp` and `icu` modules):**
  - **Demographics:** `gender`, `anchor_age`, `ethnicity`, `admission_type`
  - **Vital Signs:** `heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `temperature`, `spo2`
  - **Comprehensive Blood Chemistry / Labs:**
    - Glucose, Creatinine, BUN, Potassium, Sodium, Chloride, Bicarbonate
    - Complete Blood Count: White Blood Cells (WBC), Hemoglobin, Hematocrit, Platelets
    - Cardiac / Inflammatory Biomarkers: Troponin-T, CK-MB, C-Reactive Protein (CRP), Arterial Lactate
- **Disease / Target Labels (14 CheXpert Pathology Classes + ICD-9/10 Diagnoses):**
  - Cardiomegaly (Enlarged Heart)
  - Congestive Heart Failure / Pulmonary Edema
  - Pneumonia / Lung Consolidation
  - Pleural Effusion
  - Pneumothorax
  - Atelectasis
  - No Finding (Normal Chest)
- **Patient-Level Linkage Architecture (Exact Schema):**
  $$\begin{aligned}
  \text{MIMIC-CXR Image Metadata } (\text{dicom\_id}, \text{study\_id}, \text{subject\_id}) & \\
  \Big\downarrow & \quad [\text{Join on } \text{subject\_id} + \text{study\_id}] \\
  \text{MIMIC-IV Clinical Labs \& Vitals } (\text{labevents}, \text{chartevents}) & \\
  \Big\downarrow & \\
  \text{Same-Patient Vector: } [\text{Age, Sex, Vitals, Chemistry Labs}] & + [\text{Frontal Chest X-Ray}] \longrightarrow \text{Multimodal GNN}
  \end{aligned}$$
- **Public Accessibility & License:** Openly available to the academic community for free under the PhysioNet Credentialed Data Use Agreement. Requires completing the standard free CITI "Data or Specimens Only Research" web course (~2 hours).
- **Academic Feasibility:** **MAXIMAL GOLD STANDARD**. Pre-processed sub-cohorts (e.g., 5,000–10,000 paired patients) are standard in academic literature and easily stored locally (~2–5 GB).
- **Patient-Level Splitting:** **YES** (Standard official patient-level splits guarantee zero `subject_id` contamination between train, val, and test).
- **Scientific Classification:** **Category A: Genuine Paired Multimodal Dataset (Full Clinical Panel + Medical Images)**
- **Suitability for Multimodal GNN:** **PERFECT 10/10 FIT**. Directly satisfies the target architecture:
  - Clinical features (Vitals + Chemistry + Demographics) $\rightarrow$ `ClinicalEncoder` MLP
  - Frontal Chest X-Ray $\rightarrow$ `ImageEncoder` (ResNet18 / EfficientNet-B0)
  - Fused embedding $\rightarrow$ Patient Similarity Graph ($k$-NN) $\rightarrow$ GNN $\rightarrow$ Multi-Disease Prediction (Cardiomegaly, Heart Failure, Pneumonia) $\rightarrow$ Dual XAI (Integrated Gradients for Labs + Grad-CAM for CXR).
- **Limitations:** Requires free CITI credentialing account on PhysioNet to download the full dataset.

#### 4.2 COVID-19-NY-SBU (Stony Brook University / TCIA)
- **Official Source:** The Cancer Imaging Archive (TCIA) ([COVID-19-NY-SBU](https://doi.org/10.7937/TCIA.BB6Z-KC39))
- **Image Type:** Frontal Chest X-Rays (`.dcm` / `.png`)
- **Dataset Size:** 1,384 COVID-19 patients with **5,000+ Chest X-rays**
- **Patient Identifiers:** `PatientID` linking clinical spreadsheet to imaging studies.
- **Clinical Variables Available:** Age, Sex, BMI, Comorbidities (Diabetes, Hypertension, CKD), Vitals (SpO2, Pulse), and Lab tests (Ferritin, D-Dimer, CRP, Lymphocytes).
- **Disease / Target Labels:** Severe COVID-19 Pneumonia, ICU Admission, Need for Mechanical Ventilation, In-Hospital Mortality.
- **Scientific Classification:** **Category A: Genuine Paired Multimodal Dataset**
- **Public Accessibility:** Open access on TCIA (Creative Commons CC BY 4.0).
- **Suitability for Multimodal GNN:** **HIGH**. Good sample size (1,384 patients), full clinical labs + CXRs, but narrower disease scope (COVID-19 acute respiratory distress).

#### 4.3 NIH ChestX-ray14 & CheXpert
- **Official Source:** NIH Clinical Center / Stanford AIMI
- **Image Type:** Frontal CXRs (112,120 images / 224,316 images)
- **Patient Identifiers:** `Patient ID` present.
- **Clinical Variables Available:** `Patient Age`, `Patient Gender`, `View Position` only. **No blood chemistry, no lab panels, no vitals.**
- **Scientific Classification:** **Category B: Image-Only with Basic Demographics**
- **Suitability for Multimodal GNN:** **POOR** (Insufficient clinical features for a true multi-parametric clinical encoder).

---

## Comprehensive Dataset Comparison Matrix

| Candidate Dataset | Modality Pair | Number of Patients | Clinical Features | Medical Image Type | Patient Identifier | Linkage Type | License / Access | Suitability for Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MIMIC-CXR + MIMIC-IV** | Clinical Labs + Vitals + CXR | **65,379** | Age, Sex, BP, HR, SpO2, Glucose, Creatinine, BUN, WBC, Troponin | 2D Frontal Chest Radiograph (CXR) | `subject_id`<br>`study_id` | **Genuine Same-Patient Key** | PhysioNet Open Access (CITI Credentialed) | **10 / 10 (Highest Recommendation)** |
| **EchoNet-Dynamic** | Demographics + Vitals + Echo | **10,030** | Age, Sex, EF, EDV, ESV | 2D Apical-4-Chamber Echocardiogram | `FileName` (1-to-1 Patient ID) | **Genuine Same-Patient Key** | Stanford Research DUA (Free Click-Through) | **9 / 10 (Excellent for Cardiac)** |
| **ODIR-5K** | Demographics + Bilateral Fundus | **5,000** | Patient Age, Patient Sex, Clinical Notes | 2D Color Fundus (Left + Right Eye) | `ID` (0 to 4999) | **Genuine Same-Patient Key** | Open Access (Kaggle / Grand-Challenge) | **8 / 10 (Best for Ophthalmic)** |
| **COVID-19-NY-SBU** | Clinical Labs + Vitals + CXR | **1,384** | Age, Sex, BMI, SpO2, Ferritin, D-Dimer, CRP | 2D Frontal Chest Radiograph | `PatientID` | **Genuine Same-Patient Key** | TCIA Open Access (CC BY 4.0) | **8 / 10 (Good, Acute Focus)** |
| **APTOS 2019 / EyePACS** | Image + DR Severity | 3,662 / 88,702 | **None** | 2D Color Fundus | `id_code` / `image` | **Image-Only (No Clinical Linkage)** | Open Access (Kaggle) | **1 / 10 (Unsuitable)** |
| **Messidor-2** | Image + DR Severity | 874 | **None** | 2D Color Fundus | `image_id` | **Image-Only (No Clinical Linkage)** | Open Access | **1 / 10 (Unsuitable)** |
| **CT Kidney Dataset** | Image + Finding | Unknown | **None** | 2D CT Slices | None | **Image-Only (No Patient IDs)** | Open Access | **1 / 10 (Unsuitable)** |
| **NIH ChestX-ray14** | Image + Demographics | 30,805 | Age, Sex only (No Labs) | 2D Frontal CXR | `Patient ID` | **Image-Only + Demographics** | Open Access | **3 / 10 (Insufficient Clinicals)** |
| **UK Biobank CMR** | Full NHS EHR + Cardiac MRI | 40,000+ | Exhaustive Clinical & Genetic Panel | Multi-View Cardiac MRI | `eid` | **Genuine Same-Patient Key** | Restricted / Commercial Fee (£3k–£9k) | **4 / 10 (Inaccessible Timeline)** |

---

## Strategic Recommendation for Architecture Implementation

### Primary Recommendation: **MIMIC-CXR linked with MIMIC-IV**
- **Why this dataset is the optimal choice:**
  1. **Authentic Multimodal Clinical + Imaging Ground Truth:** It provides true clinical measurements (age, sex, systolic/diastolic blood pressure, heart rate, blood glucose, creatinine, BUN, electrolytes, blood counts) linked to high-resolution 2D medical images (Chest X-rays) for the exact same hospital encounters.
  2. **Multi-Disease Pathology Coverage:** Enables genuine multi-task prediction across cardiopulmonary conditions (**Cardiomegaly / Heart Failure**, **Pulmonary Edema**, **Pneumonia / Consolidation**, **Pleural Effusion**).
  3. **Direct Compatibility with Codebase Architecture:**
     - Clinical panel feeds directly into `ClinicalEncoder` (MLP).
     - Frontal CXR feeds directly into `ImageEncoder` (ResNet18 / EfficientNet-B0).
     - Both embeddings merge in `MultimodalFusion`.
     - Fused representations construct the $k$-NN `PatientGraphBuilder`.
     - `MultiDiseaseGNN` predicts disease risk.
     - Dual XAI: **Captum Integrated Gradients** explains clinical lab contributions, and **Grad-CAM** (`ImageGradCAM`) highlights radiographic pathology on the Chest X-ray.

### Secondary Lightweight Alternative (No Credentialing Needed): **ODIR-5K**
- If immediate open-access download without CITI credentialing is preferred:
  - **ODIR-5K** provides 5,000 patients with `Age`, `Sex`, and bilateral fundus images predicting **Diabetes**, **Hypertension**, **Glaucoma**, and **Cataract**.

---

```
========================================================================================
RESEARCH CONCLUSION:
- Real Paired Multimodal Datasets Identified: MIMIC-CXR+MIMIC-IV, EchoNet-Dynamic, ODIR-5K
- Status: Research report completed. No code changes, training, or downloads initiated.
- Next Step: Awaiting user decision on which candidate dataset to pursue.
========================================================================================
```
