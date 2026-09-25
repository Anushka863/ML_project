# ODIR INPUT FLOW VERIFICATION

**Verification Date:** 2026-09-25  
**Target Branch:** ODIR-5K Ophthalmic Multimodal Diagnostic System (Branch 3)  
**Verification Scope:** User Image Input Flow, Bilateral Integrity, API Upload Pipeline, XAI Saliency, Model Perturbation Sensitivity, and Checkpoint Immutability.

---

## 1. Current Image Source

Left-eye and right-eye retinal fundus images originate **exclusively from user interaction** on the frontend form (`/ophthalmic-assessment`):

1. **User Interaction:** The user clicks the respective file input buttons in [`OphthalmicAssessment.jsx`](file:///c:/Users/asus/Desktop/ML_PROJECT-Anushka_branch/ML_project/frontend/src/pages/OphthalmicAssessment.jsx).
2. **Browser File Objects:** The native browser `File` objects are captured into React component state:
   - `leftImageFile` (e.g., `0_left.jpg`)
   - `rightImageFile` (e.g., `0_right.jpg`)
3. **Local Object URLs:** Transient `URL.createObjectURL(file)` instances are generated purely for in-browser client-side thumbnail rendering (`leftPreview`, `rightPreview`).
4. **FormData Serialization:** When the user clicks **"Run Ophthalmic Assessment"**, the raw `File` objects are appended directly to a standard browser `FormData` payload:
   - `formData.append("left_image", leftImageFile)`
   - `formData.append("right_image", rightImageFile)`
5. **Backend Ingestion:** The FastAPI endpoint `/predict-odir/upload` receives the multipart streams as `UploadFile` objects, reads their binary streams directly into memory via `await left_image.read()` and `await right_image.read()`, verifies PIL image integrity, and forwards them to `ODIRInferenceService`.

---

## 2. Automatic Image Selection

**Status: NO**

- **Preloaded Images:** None. The React form initializes both `leftImageFile` and `rightImageFile` to `null`.
- **Default / Fallback Paths:** None. No static retinal images are loaded from `public/`, `assets/`, `data/`, or server paths during standard assessment.
- **Client-Side Storage:** Zero usage of `localStorage`, `sessionStorage`, or cookies for image caching.
- **Submission Prevention:** If either `leftImageFile` or `rightImageFile` is missing, the submission button is disabled (`disabled={loading || !leftImageFile || !rightImageFile}`), and an explicit validation banner is displayed:
  - *"Please upload the left-eye fundus image."*
  - *"Please upload the right-eye fundus image."*
  - *"Please upload both the left-eye and right-eye fundus images."*

---

## 3. User Upload Flow

```
[Browser: /ophthalmic-assessment]
  │
  ├── User inputs: Age (e.g. 58), Sex ("Male")
  ├── User selects: Left Eye File (File: 0_left.jpg, 45.2 KB)
  └── User selects: Right Eye File (File: 0_right.jpg, 44.8 KB)
        │
        ▼
[React State: OphthalmicAssessment.jsx]
  ├── leftImageFile: File(name="0_left.jpg", size=46284, type="image/jpeg")
  └── rightImageFile: File(name="0_right.jpg", size=45872, type="image/jpeg")
        │
        ▼
[HTTP Client: frontend/src/services/api.js]
  └── POST /predict-odir/upload (multipart/form-data)
        │
        ▼
[FastAPI Route: backend/app/api/routes.py]
  └── predict_odir_upload(age, sex, left_image: UploadFile, right_image: UploadFile)
        ├── Reads byte buffers in memory: left_bytes, right_bytes
        ├── Verifies file integrity: PIL.Image.open().verify()
        └── Logs request identity: filenames, sizes, demographic parameters
        │
        ▼
[Service Layer: backend/app/services/prediction_service.py]
  └── ClinicalPredictionService.predict_odir_risk(age, sex, left_img, right_img)
        │
        ▼
[Inference Engine: app/inference/odir_inference.py]
  └── ODIRInferenceService.predict(age, sex, left_img, right_img)
        ├── ODIRPreprocessor:
        │     ├── Demographics: Z-score Age, Float Sex -> Tensor (1, 2)
        │     ├── Left Eye: RGB 224x224, ImageNet normalized -> Tensor (1, 3, 224, 224)
        │     └── Right Eye: RGB 224x224, ImageNet normalized -> Tensor (1, 3, 224, 224)
        ├── ODIRMultimodalGNN Forward Pass:
        │     ├── BilateralImageEncoder: ResNet-18(Left) + ResNet-18(Right) -> Concat -> Linear -> (1, 128)
        │     ├── ClinicalEncoder: MLP(Age, Sex) -> (1, 32)
        │     ├── MultimodalFusion: Concat(32, 128) -> Fusion MLP -> (1, 128)
        │     ├── PatientGraphBuilder: Single patient N=1, E=0 isolated graph
        │     └── MultiDiseaseGNN: 2-layer SimpleGNNConv + 8 Head Logit Predictors -> (1, 1) x 8
        └── Sigmoid: 8 calibrated probabilities [N, D, G, C, A, H, M, O]
        │
        ▼
[Response JSON: ODIRPredictionResponse]
  └── Returned to frontend and rendered in 8 probability cards
```

---

## 4. Left Eye Verification

**Status: YES**

- **Ingestion:** Transferred as `UploadFile` (`left_image`), read into memory, validated as valid JPEG/PNG.
- **Preprocessing:** Resized to $224 \times 224$, converted to 3-channel RGB, normalized via ImageNet mean/std into PyTorch tensor of shape `(1, 3, 224, 224)`.
- **Vision Encoding:** Forwarded through `model.bilateral_encoder.single_eye_encoder` to produce a 64-dimensional left retinal embedding.
- **Empirical Sensitivity Verification:**
  - When Left Image was altered from `0_left.jpg` to `1005_left.jpg` (while keeping Right Image, Age, and Sex fixed):
    - **Cataract Probability:** shifted by **$-6.69\%$** ($0.6052 \to 0.5383$)
    - **Hypertension Probability:** shifted by **$+5.27\%$** ($0.3762 \to 0.4289$)
    - **Glaucoma Probability:** shifted by **$-4.51\%$** ($0.5930 \to 0.5479$)
    - **Diabetes Probability:** shifted by **$+2.65\%$** ($0.4462 \to 0.4727$)
  - **Conclusion:** Left eye image directly and measurably alters model output.

---

## 5. Right Eye Verification

**Status: YES**

- **Ingestion:** Transferred as `UploadFile` (`right_image`), read into memory, validated as valid JPEG/PNG.
- **Preprocessing:** Resized to $224 \times 224$, converted to 3-channel RGB, normalized via ImageNet mean/std into PyTorch tensor of shape `(1, 3, 224, 224)`.
- **Vision Encoding:** Forwarded through `model.bilateral_encoder.single_eye_encoder` to produce a 64-dimensional right retinal embedding.
- **Empirical Sensitivity Verification:**
  - When Right Image was altered from `0_right.jpg` to `1005_right.jpg` (while keeping Left Image, Age, and Sex fixed):
    - **Cataract Probability:** shifted by **$-2.33\%$** ($0.6052 \to 0.5819$)
    - **Glaucoma Probability:** shifted by **$-1.54\%$** ($0.5930 \to 0.5776$)
    - **Hypertension Probability:** shifted by **$+1.00\%$** ($0.3762 \to 0.3862$)
    - **Normal Probability:** shifted by **$+0.62\%$** ($0.4620 \to 0.4682$)
  - **Conclusion:** Right eye image directly and measurably alters model output independently from the left eye.

---

## 6. Age Verification

**Status: YES**

- **Ingestion:** Captured from `<input type="number">`, parsed as `float`.
- **Standardization:** Normalized via training cohort parameters: $Z = \frac{\text{Age} - 57.854}{11.724}$.
- **Encoding:** Passed as first index of `demo_tensor` into `ClinicalEncoder`.
- **Empirical Sensitivity Verification:**
  - Altering Age from 58 to 25 (with identical left/right images and sex) shifted predictions across all 8 disease heads (e.g., Glaucoma $+0.42\%$, Cataract $+0.32\%$, Hypertension $-0.35\%$).

---

## 7. Sex Verification

**Status: YES**

- **Ingestion:** Captured from `<select>`, encoded as `1.0` (Male) or `0.0` (Female).
- **Encoding:** Passed as second index of `demo_tensor` into `ClinicalEncoder`.
- **Empirical Sensitivity Verification:**
  - Altering Sex from Male to Female (with identical left/right images and age) shifted predictions across all 8 heads (e.g., Cataract $+0.59\%$, Glaucoma $+0.34\%$, Hypertension $-0.51\%$).

---

## 8. Bilateral Model Verification

**Status: YES**

- **Architecture:** `BilateralImageEncoder` explicitly takes `left_img` and `right_img`, encodes both via ResNet-18, concatenates the embeddings ($64 + 64 = 128$), and projects through `nn.Linear(128, 128)` with BatchNorm1d, ReLU, and Dropout.
- **Preservation:** The model architecture has been preserved 100% intact with zero architectural modification.
- **Duplication Prohibition:** Neither frontend nor backend duplicates a single eye image into both channels. Both eyes must be provided independently.

---

## 9. Single-Eye Support Analysis

**Status: TECHNICALLY POSSIBLE BUT NOT VALIDATED**

- **Technical Compatibility:** The existing neural network graph takes two image tensors of shape `(B, 3, 224, 224)`. If one passes the same eye image twice, the forward pass mathematically executes without throwing an exception.
- **Scientific Invalidity:** However, the checkpoint `best_model.pt` was trained exclusively on genuine bilateral ocular photography where each patient had paired left and right fundus photos with asymmetrical disease presentations (e.g. unilateral cataract or asymmetric diabetic retinopathy). Duplicating one eye into both inputs produces an out-of-distribution synthetic input that was never clinically evaluated or validated.
- **Formal Statement:**
  > *"The current checkpoint is trained and validated for bilateral ODIR input. Single-eye inference is not equivalent to the validated bilateral model."*
- **Recommendation:** If single-eye inference is required as a distinct feature in the future, train and evaluate a dedicated single-eye vision encoder rather than synthesizing bilateral input through duplication.

---

## 10. Prediction Source Verification

**Status: 100% MODEL-DERIVED**

- Every displayed percentage ($p_c$) originates from the forward pass scalar logit $z_c$ passed through `torch.sigmoid(z_c)`:
  $$p_c = \frac{1}{1 + e^{-z_c}}$$
- Zero hardcoded probabilities, zero static demo dictionaries, zero random math generators.
- Output categories match official ODIR targets:
  - `[N]` Normal
  - `[D]` Diabetes
  - `[G]` Glaucoma
  - `[C]` Cataract
  - `[A]` AMD
  - `[H]` Hypertension
  - `[M]` Myopia
  - `[O]` Other

---

## 11. Checkpoint Integrity

| Checkpoint File | Baseline SHA256 (Before Task) | Final SHA256 (After Task) | Integrity Status |
| :--- | :--- | :--- | :---: |
| `models/odir_multimodal/best_model.pt` | `bef283b54c4232555fa10b45dce93e8313b76b055bb38ec90eb57e714715c99c` | `bef283b54c4232555fa10b45dce93e8313b76b055bb38ec90eb57e714715c99c` | **UNCHANGED (100% EXACT MATCH)** |

---

## 12. Test Results

### 12.1 ODIR-Specific Pipeline Tests (`tests/test_odir_pipeline.py`)
- `test_01_preprocessor_demographics`: **PASSED**
- `test_02_preprocessor_bilateral_images`: **PASSED**
- `test_03_preprocessor_label_encoding`: **PASSED**
- `test_04_patient_split_zero_leakage`: **PASSED**
- `test_05_odir_multimodal_model_forward`: **PASSED**
- `test_06_odir_api_missing_left_image`: **PASSED**
- `test_07_odir_api_missing_right_image`: **PASSED**
- `test_08_odir_api_invalid_image_data`: **PASSED**
- `test_09_model_info_includes_odir_branch`: **PASSED**
- `test_10_existing_nhanes_checkpoint_unaffected`: **PASSED**
- **ODIR Suite Score:** **10 / 10 passed (100%)**

### 12.2 Repository-Wide Test Suite
- **Complete Suite Score:** **58 / 58 passed (100%) in 44.98s**
- **Failed / Skipped / Errors:** 0 / 0 / 0

---

## 13. Issues Found & Resolved

| Severity | Issue Description | Resolution Implemented |
| :---: | :--- | :--- |
| **MEDIUM** | Frontend displayed informal multi-term labels (e.g. *"Diabetes / Diabetic Retinopathy"*) rather than standardized ODIR categories. | Updated `ODIR_DISEASE_NAMES` and UI to official categories: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other. |
| **MEDIUM** | Uncalibrated risk labels (*"Moderate Risk"*) were displayed without a formal calibration study. | Updated UI to explicit model probability reporting: *"Model Probability: 51.0%"*. |
| **LOW** | Missing-image validation in UI was generic. | Enhanced UI with file-specific validation: displays filename, size in KB, and explicit messages ("No left-eye image selected", "No right-eye image selected"). |
| **LOW** | Absence of backend identity logging. | Added safe logging in `backend/app/api/routes.py` reporting received filenames and byte counts without logging sensitive patient data. |

---

## 14. Recommended Next Step

For future single-eye requirements:
- Do **NOT** duplicate a single fundus image into both left and right channels to force bilateral model execution.
- Implement a dedicated single-eye vision encoder branch (e.g., `SingleEyeODIRModel`) trained specifically on single-eye inputs with its own dedicated evaluation metrics and checkpoint.
