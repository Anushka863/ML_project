# PTB-XL Multimodal GNN Training Report

**Model Architecture**: `PTBXLMultimodalGNN` (`ClinicalEncoder` + `ECGEncoder` + `MultimodalFusion` + `PatientGraphBuilder` + `MultiDiseaseGNN`)  
**Execution Mode**: Full Training Run  
**Hardware Device**: CPU  
**Random Seed**: 42  
**Date**: 2026-09-20 12:51:15  

---

## 1. Dataset & Patient-Level Splitting
- **Total Dataset Records**: 21799
- **Total Unique Patients**: 18869
- **Training Set**: 13208 unique patients (15208 / 15208 records used)
- **Validation Set**: 2830 unique patients (3319 / 3319 records used)
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
- **Loss Function**: `BCEWithLogitsLoss` (pos_weight=0.79)
- **Optimizer**: `AdamW` (learning_rate=0.001, weight_decay=0.0001)
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2)
- **Batch Size**: 64
- **Early Stopping Patience**: 5 epochs

---

## 3. Best Validation Performance (Epoch 5)
- **Best Validation Loss**: 0.3131
- **Validation Accuracy**: 84.66%
- **Validation Precision**: 0.8726
- **Validation Recall / Sensitivity**: 0.8556
- **Validation F1-Score**: 0.8640
- **Validation ROC-AUC**: 0.9223

---

## 4. Epoch Metrics Summary

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | LR |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3977 | 0.3972 | 0.7785 | 0.9522 | 0.6434 | 0.7679 | 0.9126 | 1.0e-03 |
| 2 | 0.3315 | 0.3722 | 0.7999 | 0.9527 | 0.6825 | 0.7953 | 0.9167 | 1.0e-03 |
| 3 | 0.3201 | 0.3201 | 0.8448 | 0.8730 | 0.8513 | 0.8620 | 0.9210 | 1.0e-03 |
| 4 | 0.3199 | 0.3296 | 0.8132 | 0.9434 | 0.7148 | 0.8134 | 0.9163 | 1.0e-03 |
| 5 | 0.3137 | 0.3131 | 0.8466 | 0.8726 | 0.8556 | 0.8640 | 0.9223 | 1.0e-03 |

---

## 5. Generated Artifacts & Checkpoints
- **Best Multimodal Model Checkpoint**: `models/ptbxl_multimodal/best_model.pt`
- **ECG Encoder Checkpoint**: `models/ptbxl_multimodal/ecg_encoder.pt`
- **Training Metrics CSV**: `reports/training_metrics.csv`
- **Training Curves Plot**: `reports/ptbxl_training_curves.png`
