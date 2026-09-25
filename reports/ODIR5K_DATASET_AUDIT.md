# ODIR-5K Dataset Read-Only Audit Report

STATUS: DATASET PARTIALLY SUITABLE

---

## 1. Dataset Location

- **Primary Archive Directory:** `ML_project/data/odir5k/ODIR-5K/ODIR-5K/` (and root `data/odir5k/ODIR-5K/ODIR-5K/`)
- **Metadata Spreadsheet:** `ML_project/data/odir5k/ODIR-5K/ODIR-5K/data.xlsx`
- **Training Image Directory:** `ML_project/data/odir5k/ODIR-5K/ODIR-5K/Training Images/`
- **Testing Image Directory:** `ML_project/data/odir5k/ODIR-5K/ODIR-5K/Testing Images/`
- **Preprocessed Image Cache:** `ML_project/data/odir5k/preprocessed_images/` (6,392 images)
- **Consolidated Metadata CSV:** `ML_project/data/odir5k/full_df.csv` (6,392 records)

---

## 2. Dataset Structure

The physical directory tree verified locally on disk is structured as follows:

```
data/odir5k/
├── full_df.csv                                      # Pre-extracted tabular metadata (6,392 rows)
├── preprocessed_images/                             # Preprocessed fundus image cache (6,392 .jpg files)
└── ODIR-5K/
    └── ODIR-5K/
        ├── data.xlsx                                # Master Ground-Truth Metadata Spreadsheet (1.3 MB)
        ├── Training Images/                         # 7,000 fundus image files (.jpg)
        └── Testing Images/                          # 1,000 fundus image files (.jpg)
```

- **Folder Structure Summary:**
  - `Training Images/`: Subdirectory containing all 7,000 paired training images for 3,500 patients.
  - `Testing Images/`: Subdirectory containing 1,000 paired test images for 500 challenge evaluation patients.
  - `data.xlsx`: Official Peking University ODIR-5K challenge annotations for the 3,500 training patients.

---

## 3. Image Statistics

| Metric | Training Images | Testing Images | Preprocessed Cache | Combined Total |
| :--- | :--- | :--- | :--- | :--- |
| **Total Image Files** | 7,000 | 1,000 | 6,392 | 8,000 raw / 6,392 cached |
| **File Formats / Extensions** | 100% `.jpg` | 100% `.jpg` | 100% `.jpg` | 100% `.jpg` |
| **Corrupted / Zero-byte Files** | 0 | 0 | 0 | 0 |
| **Left-Eye Images** | 3,500 (`*_left.jpg`) | 500 (`*_left.jpg`) | 3,196 (`*_left.jpg`) | 4,000 raw |
| **Right-Eye Images** | 3,500 (`*_right.jpg`)| 500 (`*_right.jpg`)| 3,196 (`*_right.jpg`)| 4,000 raw |
| **Filename Pattern Adherence** | 100% (`^\d+_(left\|right)\.jpg$`) | 100% (`^\d+_(left\|right)\.jpg$`) | 100% | **100.0% Strict Conformance** |

- **Image Filename Pattern Verification:**
  - Standard training format: `0_left.jpg`, `0_right.jpg`, `1_left.jpg`, `1_right.jpg`, ..., `4784_right.jpg`.
  - Standard testing format: `1000_left.jpg`, `1000_right.jpg`, `1001_left.jpg`, `1001_right.jpg`, ..., `4794_right.jpg`.
  - Exactly 100% of images in both directories adhere to the strict regex `<PatientID>_<left|right>.jpg`.

---

## 4. Metadata Statistics (`data.xlsx`)

- **Total Rows:** 3,500 patient records.
- **Total Columns:** 15 columns.
- **All 15 Column Names:**
  1. `ID`
  2. `Patient Age`
  3. `Patient Sex`
  4. `Left-Fundus`
  5. `Right-Fundus`
  6. `Left-Diagnostic Keywords`
  7. `Right-Diagnostic Keywords`
  8. `N` (Normal)
  9. `D` (Diabetes)
  10. `G` (Glaucoma)
  11. `C` (Cataract)
  12. `A` (Age-related Macular Degeneration)
  13. `H` (Hypertension)
  14. `M` (Pathological Myopia)
  15. `O` (Other diseases / abnormalities)

### Data Types & Missing Values Audit

| Column Name | Inferred Type | Python Data Type | Missing Count | % Missing | Unique Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ID` | Integer ID | `int64` | 0 | 0.0% | 3,500 |
| `Patient Age` | Discrete Numeric | `int64` | 0 | 0.0% | 76 (range: 1 – 91, mean: 57.85) |
| `Patient Sex` | Categorical | `object` / `str` | 0 | 0.0% | 2 ('Male': 1,885, 'Female': 1,615) |
| `Left-Fundus` | Filename String | `str` | 0 | 0.0% | 3,500 |
| `Right-Fundus` | Filename String | `str` | 0 | 0.0% | 3,500 |
| `Left-Diagnostic Keywords` | Text / Clinical Annotation | `str` | 0 | 0.0% | Free text clinical notes |
| `Right-Diagnostic Keywords`| Text / Clinical Annotation | `str` | 0 | 0.0% | Free text clinical notes |
| `N` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `D` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `G` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `C` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `A` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `H` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `M` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |
| `O` | Binary Indicator | `int64` | 0 | 0.0% | 2 (0 or 1) |

### First 5 Rows of `data.xlsx`

```
Row 0:
  ID: 0 | Age: 69 | Sex: Female | Left-Fundus: 0_left.jpg | Right-Fundus: 0_right.jpg
  Keywords: Left="cataract" | Right="normal fundus"
  Labels: N=0, D=0, G=0, C=1, A=0, H=0, M=0, O=0

Row 1:
  ID: 1 | Age: 57 | Sex: Male | Left-Fundus: 1_left.jpg | Right-Fundus: 1_right.jpg
  Keywords: Left="normal fundus" | Right="normal fundus"
  Labels: N=1, D=0, G=0, C=0, A=0, H=0, M=0, O=0

Row 2:
  ID: 2 | Age: 42 | Sex: Male | Left-Fundus: 2_left.jpg | Right-Fundus: 2_right.jpg
  Keywords: Left="laser spots,..." | Right="moderate non proliferative retinopathy"
  Labels: N=0, D=1, G=0, C=0, A=0, H=0, M=0, O=1

Row 3:
  ID: 3 | Age: 66 | Sex: Male | Left-Fundus: 3_left.jpg | Right-Fundus: 3_right.jpg
  Keywords: Left="dry age-related macular degeneration" | Right="dry age-related macular degeneration"
  Labels: N=0, D=0, G=0, C=0, A=1, H=0, M=0, O=1

Row 4:
  ID: 4 | Age: 53 | Sex: Male | Left-Fundus: 4_left.jpg | Right-Fundus: 4_right.jpg
  Keywords: Left="macular epiretinal membrane" | Right="mild nonproliferative retinopathy"
  Labels: N=0, D=1, G=0, C=0, A=0, H=0, M=0, O=1
```

---

## 5. Patient ID Analysis

- **Unique Patient IDs:** Exactly 3,500 unique patient IDs in `data.xlsx`.
- **ID Numbering Scheme:** Non-contiguous integer IDs ranging from `0` to `4784` in the training set; the remaining 500 IDs up to `4794` are allocated to the testing set.
- **Row-to-ID Relationship:** Each row represents exactly one unique human subject.
- **ID Integrity:** No duplicate IDs, no null IDs, and no synthetic/fabricated IDs exist in the source metadata.

---

## 6. Image-to-Patient Linkage Audit

| Linkage Dimension | Audited Count | Percentage |
| :--- | :--- | :--- |
| **Total Metadata Patient Records** | 3,500 | 100.0% |
| **Valid Left-Eye Image Matches (`Training Images/`)** | 3,500 | 100.0% |
| **Valid Right-Eye Image Matches (`Training Images/`)** | 3,500 | 100.0% |
| **Missing Left-Eye Images** | 0 | 0.0% |
| **Missing Right-Eye Images** | 0 | 0.0% |
| **Bilateral Paired Image Matches (Both Left & Right Exist)** | 3,500 | 100.0% |

- **Linkage Mechanism:**
  - Linkage is **explicit and deterministic** via filename string references in `Left-Fundus` and `Right-Fundus` columns, which match physical files on disk:
    - Patient `0` $\rightarrow$ `Left-Fundus = "0_left.jpg"`, `Right-Fundus = "0_right.jpg"`.
    - Patient `1005` $\rightarrow$ `Left-Fundus = "1005_left.jpg"`, `Right-Fundus = "1005_right.jpg"`.
  - Linkage is **NOT** inferred from row position or file sorting order.
  - Zero missing files or broken references were detected across the entire 3,500-patient training cohort.

---

## 7. Clinical Feature Analysis

A systematic audit of tabular clinical features was conducted to assess feature richness:

### Available Features (Genuinely Present in ODIR-5K)
- `Patient Age`: Continuous integer (1 to 91 years; Mean: 57.85 $\pm$ 11.72 years).
- `Patient Sex`: Categorical ('Male' = 1,885, 'Female' = 1,615).
- `Diagnostic Keywords`: Free-text clinical diagnosis per eye.

### Not Available Features (Completely Absent from ODIR-5K)
- `Glucose` / Fasting Blood Sugar: **NOT AVAILABLE**
- `HbA1c` (Glycated Hemoglobin): **NOT AVAILABLE**
- `Creatinine`: **NOT AVAILABLE**
- `BUN` (Blood Urea Nitrogen): **NOT AVAILABLE**
- `Blood Pressure` (Systolic BP / Diastolic BP): **NOT AVAILABLE**
- `HDL Cholesterol`: **NOT AVAILABLE**
- `Total Cholesterol`: **NOT AVAILABLE**
- `BMI` (Body Mass Index): **NOT AVAILABLE**
- `Waist Circumference`: **NOT AVAILABLE**
- `eGFR` (Estimated Glomerular Filtration Rate): **NOT AVAILABLE**

---

## 8. Disease Label Analysis

The ODIR-5K dataset provides multi-label diagnostic ground truth across 8 standard ophthalmic disease categories:

| Code | Disease Category Name | Positive Cases | Prevalence (%) | Clinical Meaning & Pathology |
| :---: | :--- | :---: | :---: | :--- |
| **N** | **Normal** | 1,140 | 32.57% | Normal ocular fundus without signs of retinopathy or maculopathy. |
| **D** | **Diabetes / Diabetic Retinopathy** | 1,128 | 32.23% | Retinal vascular damage from diabetes: microaneurysms, hemorrhages, hard/soft exudates, neovascularization. |
| **G** | **Glaucoma** | 215 | 6.14% | Glaucomatous optic neuropathy characterized by enlarged cup-to-disc ratio and neuroretinal rim thinning. |
| **C** | **Cataract** | 212 | 6.06% | Crystalline lens opacification causing hazy/blurred fundus photography. |
| **A** | **Age-related Macular Degeneration (AMD)** | 164 | 4.69% | Macular drusen, geographic atrophy (dry AMD), or choroidal neovascular membrane (wet AMD). |
| **H** | **Hypertension / Hypertensive Retinopathy** | 103 | 2.94% | Retinal vascular changes from chronic high BP: arteriolar narrowing, arteriovenous nicking, flame hemorrhages. |
| **M** | **Pathological Myopia** | 174 | 4.97% | High myopia degenerative changes: posterior staphyloma, chorioretinal atrophy, lacquer cracks. |
| **O** | **Other Diseases / Abnormalities** | 979 | 27.97% | Other ocular conditions: epiretinal membrane, retinal vein occlusion (RVO), macular hole, retinal detachment, laser scars. |

*Note: Total positive labels sum to 4,115 across 3,500 patients because ODIR is a multi-label classification dataset where patients can have co-occurring ocular conditions (e.g., Diabetic Retinopathy + Hypertension or Cataract + AMD).*

---

## 9. NHANES Compatibility

- **Direct Linkage Possibility:** **STRICTLY NO.**
- **Reasoning:**
  - **Cohorts:** NHANES is a continuous United States population health survey conducted by the US CDC / NCHS. ODIR-5K is an ophthalmic hospital challenge cohort collected across multiple medical centers in China (Peking University, Shanggong Medical Technology).
  - **Identifiers:** There is zero overlap in patient identification numbers (`SEQN` in NHANES vs `ID` in ODIR-5K).
  - **Scientific Integrity:** Matching ODIR fundus images with NHANES clinical records based on age, sex, row order, or nearest-neighbor similarity is scientifically fraudulent and constitutes artificial data leakage.

---

## 10. PTB-XL Compatibility

- **Direct Linkage Possibility:** **STRICTLY NO.**
- **Reasoning:**
  - PTB-XL is an Austrian clinical electrocardiology database (12-lead ECG waveforms collected by Physikalisch-Technische Bundesanstalt).
  - ODIR-5K is a Chinese retinal fundus photography database.
  - The patient cohorts are completely disjoint with zero shared biological or clinical identifiers.

---

## 11. Existing Architecture Compatibility

The existing project codebase contains modular components designed for multimodal processing:

1. **`ClinicalEncoder` (`app/models/clinical_encoder.py`):**
   - Multi-layer perceptron mapping tabular clinical features ($d_{\text{in}} \rightarrow [128, 64] \rightarrow 64$).
   - Can encode ODIR demographics (`Age` and `Sex`, $d_{\text{in}}=2$) into a 64-dimensional clinical embedding.
2. **`ImageEncoder` (`app/models/image_encoder.py`):**
   - CNN backbone (ResNet-18, ResNet-50, or EfficientNet-B0) with projection head mapping $3 \times 224 \times 224$ images into 64-dimensional embeddings.
   - Fully compatible with ODIR left and right retinal fundus images.
3. **`MultimodalFusion` (`app/models/multimodal_model.py`):**
   - Concatenates clinical and image embeddings ($64 + 64 = 128 \rightarrow 256 \rightarrow 128$) to create unified multimodal representations.
4. **`PatientGraphBuilder` (`app/graph/patient_graph.py`):**
   - Builds $k$-NN patient similarity graphs using cosine similarity over multimodal embeddings, yielding edge indices and similarity weights.
5. **`MultiDiseaseGNN` (`app/models/gnn_model.py`):**
   - Message-passing neural network with multi-head classification layers (`heads = ModuleDict`).
   - Can be configured with 8 binary heads corresponding to ODIR targets (`N`, `D`, `G`, `C`, `A`, `H`, `M`, `O`).
6. **`ImageGradCAM` (`app/explainability/image_xai.py`):**
   - Computes activation heatmaps over CNN feature maps, highlighting retinal lesions (microaneurysms, hemorrhages, cup-to-disc ratio changes, drusen).

---

## 12. Multimodal Suitability

- **Bilateral Ocular Multimodal Modeling:** ODIR provides genuine paired data for multimodal representation learning:
  $$\text{ODIR Patient} = \{ \text{Left Fundus Image}, \text{Right Fundus Image}, \text{Age}, \text{Sex}, \text{8-Class Diagnostic Target} \}$$
- **Multi-View Fundus Integration:** Left and right eyes can be processed via dual-stream `ImageEncoder` or averaged feature pooling to capture bilateral systemic manifestations (e.g., bilateral diabetic retinopathy vs unilateral retinal vein occlusion).
- **Patient Similarity Graph:** A $k$-NN graph over fused fundus+demographic embeddings can facilitate transductive and inductive GNN message passing for rare ocular pathology detection (e.g., Pathological Myopia or Hypertensive Retinopathy).

---

## 13. Limitations

1. **Absence of Laboratory Biomarkers:** ODIR does **not** contain blood glucose, HbA1c, lipid profiles, blood pressure measurements, or renal markers.
2. **Cannot Replace NHANES:** ODIR **cannot** substitute or replace the existing NHANES clinical model for cardiometabolic risk prediction (diabetes, coronary heart disease, chronic kidney disease).
3. **Domain Specificity:** ODIR predictions are strictly ophthalmic (ocular manifestations and systemic diseases visible in retinal microvasculature, such as Diabetic Retinopathy and Hypertensive Retinopathy), rather than general cardiometabolic panels.

---

## 14. Final Dataset Classification & Recommended Next Steps

### Final Classification
**B. Image + limited patient-level metadata dataset**

- **Justification:** ODIR-5K contains 8,000 real medical images genuinely paired with 3,500 real human patients, complete with bilateral fundus imaging, age, sex, and expert multi-disease ground truth labels. However, it lacks comprehensive laboratory/clinical chemistry panels, restricting its tabular modality to demographic features.

### Recommended Next Step

**Implement a Standalone Ophthalmic Multimodal Branch (`ODIR Branch`) without modifying or merging NHANES/PTB-XL branches:**

1. **Retain Existing Independent Branches:**
   - **Branch 1 (Cardiometabolic Tabular CDS):** NHANES dataset $\rightarrow$ 10 clinical lab features $\rightarrow$ `ClinicalEncoder` $\rightarrow$ `MultiDiseaseGNN` (Diabetes, Heart Disease, CKD).
   - **Branch 2 (Electrophysiological Multimodal CDS):** PTB-XL dataset $\rightarrow$ 12-lead ECG waveforms + 4 demographics $\rightarrow$ `ECGEncoder` + `ClinicalEncoder` $\rightarrow$ `MultimodalFusion` $\rightarrow$ `MultiDiseaseGNN`.
2. **Add New Branch 3 (Ophthalmic Multimodal CDS):**
   - ODIR-5K dataset $\rightarrow$ Bilateral Fundus Images + Age & Sex $\rightarrow$ `ImageEncoder` (ResNet-18) + `ClinicalEncoder` $\rightarrow$ `MultimodalFusion` $\rightarrow$ `PatientGraphBuilder` $\rightarrow$ `MultiDiseaseGNN` (8 Heads: `N, D, G, C, A, H, M, O`) $\rightarrow$ `ImageGradCAM` retinal heatmap explainability.
3. **Tri-Modal Clinical Dashboard:**
   - Expose all three specialized clinical decision support modalities as distinct triage workflows within the system.
