# ODIR-5K Ophthalmic Multimodal Training Report

**Date:** 2026-09-23 23:42:07  
**Model Architecture:** ODIRMultimodalGNN (ResNet-18 Bilateral Vision + Demographic MLP + k-NN Patient Graph + Multi-Head GNN)  
**Checkpoint Path:** `C:\Users\asus\Desktop\ML_PROJECT-Anushka_branch\ML_project\models\odir_multimodal\best_model.pt`  
**Training Set:** 2450 patients (4,900 images)  
**Validation Set:** 525 patients (1,050 images)  
**Test Set:** 525 patients (1,050 images)  

---

## 1. Test Set Evaluation Summary

- **Macro AUROC:** `0.6863`
- **Macro F1 Score:** `0.2102`
- **Multi-Label Subset Accuracy:** `0.0171`

---

## 2. Per-Disease Performance Breakdown

| Target Code | Disease Condition Name | Positive Cases (Test) | AUROC | Precision | Recall | F1 Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **N** | Normal | 155 | **0.5129** | 0.3177 | 0.5677 | 0.4074 |
| **D** | Diabetes / Diabetic Retinopathy | 180 | **0.5771** | 0.4114 | 0.7611 | 0.5341 |
| **G** | Glaucoma | 31 | **0.8265** | 0.1225 | 0.8065 | 0.2128 |
| **C** | Cataract | 31 | **0.9239** | 0.1883 | 0.9355 | 0.3135 |
| **A** | Age-related Macular Degeneration (AMD) | 31 | **0.5631** | 0.0000 | 0.0000 | 0.0000 |
| **H** | Hypertension / Hypertensive Retinopathy | 14 | **0.6798** | 0.0000 | 0.0000 | 0.0000 |
| **M** | Pathological Myopia | 21 | **0.8639** | 0.1205 | 0.9524 | 0.2139 |
| **O** | Other Ocular Abnormalities | 150 | **0.5433** | 0.0000 | 0.0000 | 0.0000 |

---

## 3. Training Configuration & Hyperparameters

- **Visual Backbone:** Pretrained ResNet-18 (Bilateral dual-stream, 64-dim per eye $\rightarrow$ 128-dim bilateral projection)
- **Clinical Demographics:** Age + Sex ($2 \rightarrow 32$-dim ClinicalEncoder MLP)
- **Multimodal Fusion:** $32 + 128 \rightarrow 128$-dim patient representation
- **Patient Graph:** $k$-NN graph ($k=5$, Cosine Similarity)
- **Loss Function:** `BCEWithLogitsLoss` with positive class weighting
- **Optimizer:** AdamW (`lr=0.0001`, `weight_decay=1e-4`)
- **Batch Size:** 16
- **Epochs Trained:** 3 (Best epoch: 2)

---

## 4. Scientific Compliance
- **Patient Isolation:** Disjoint patient splitting verified (0% overlap between train, val, test).
- **Bilateral Integrity:** Left and Right eyes maintained within the same patient records.
- **Independence:** No synthetic linking or data fusion with NHANES or PTB-XL cohorts.
