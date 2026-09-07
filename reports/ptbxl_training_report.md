# PTB-XL Multimodal GNN Training Report

**Model Architecture**: `PTBXLMultimodalGNN` (`ClinicalEncoder` + `ECGEncoder` + `MultimodalFusion` + `PatientGraphBuilder` + `MultiDiseaseGNN`)  
**Execution Mode**: Full Training Run  
**Hardware Device**: CPU  
**Random Seed**: 42  
**Date**: 2026-09-05 02:51:25  

---

## 1. Dataset & Patient-Level Splitting
- **Total Dataset Records**: 21799
- **Total Unique Patients**: 18869
- **Training Set**: 13208 unique patients (2000 / 15208 records used)
- **Validation Set**: 2830 unique patients (500 / 3319 records used)
- **Test Set (Untouched)**: 2831 unique patients (3272 records reserved)
- **Class Distribution (Train)**: 6711 Normal (44.1%) / 8497 Abnormal (55.9%)
- **Split Integrity Verification**:
  - Train-Val Patient Overlap: 0 (Zero Leakage)
  - Train-Test Patient Overlap: 0 (Zero Leakage)
  - Val-Test Patient Overlap: 0 (Zero Leakage)
  - Split Status: **PASS (Patient-Disjoint Partitions Confirmed)**

---

## 2. Model Architecture & Hyperparameters
- **Clinical Feature Inputs**: 4 features (Age, Sex, Height, Weight) -> `ClinicalEncoder` (64-dim)
- **ECG Waveform Inputs**: 12 leads x 1000 timepoints (100Hz 10s) -> `ECGEncoder` 1D CNN (128-dim)
- **Multimodal Fusion**: Cross-modality projection -> `MultimodalFusion` (128-dim)
- **Graph Topology**: `PatientGraphBuilder` (k=5 nearest neighbors via Cosine Similarity)
- **GNN Backbone**: `MultiDiseaseGNN` (64 hidden -> 32 output -> binary disease logits)
- **Loss Function**: `BCEWithLogitsLoss` (pos_weight=1.15)
- **Optimizer**: `AdamW` (learning_rate=0.001, weight_decay=0.0001)
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2)
- **Batch Size**: 16
- **Early Stopping Patience**: 3 epochs

---

## 3. Best Validation Performance (Epoch 1)
- **Best Validation Loss**: 0.5913
- **Validation Accuracy**: 75.00%
- **Validation Precision**: 0.7500
- **Validation Recall / Sensitivity**: 0.6953
- **Validation F1-Score**: 0.7216
- **Validation ROC-AUC**: 0.8005

---

## 4. Epoch Metrics Summary

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | LR |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6427 | 0.5913 | 0.7500 | 0.7500 | 0.6953 | 0.7216 | 0.8005 | 1.0e-03 |
| 2 | 0.6265 | 0.6059 | 0.7420 | 0.7281 | 0.7124 | 0.7202 | 0.7862 | 1.0e-03 |
| 3 | 0.5697 | 0.5979 | 0.7480 | 0.7257 | 0.7382 | 0.7319 | 0.8011 | 1.0e-03 |
| 4 | 0.6194 | 0.7269 | 0.6180 | 0.5536 | 0.9313 | 0.6944 | 0.7827 | 5.0e-04 |

---

## 5. Generated Artifacts & Checkpoints
- **Best Multimodal Model Checkpoint**: `models/ptbxl_multimodal/best_model.pt`
- **ECG Encoder Checkpoint**: `models/ptbxl_multimodal/ecg_encoder.pt`
- **Training Metrics CSV**: `reports/training_metrics.csv`
- **Training Curves Plot**: `reports/ptbxl_training_curves.png`
