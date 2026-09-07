# Final Evaluation Report — PTB-XL Multimodal GNN

**Target**: Binary Heart Disease Detection (Normal vs Abnormal)  
**Evaluation Set**: Untouched Patient Test Set (3272 recordings, 2831 unique patients)  
**Hardware Device**: CPU  
**Evaluation Time**: 41.56s  
**Date**: 2026-09-05 02:54:03  

---

## 1. Quantitative Performance Metrics

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **Accuracy** | **72.71%** | Overall correct classification rate across test cohort |
| **Precision** | **0.8628** | True abnormal proportion among positive alarms |
| **Recall / Sensitivity** | **0.6296** | Cardiac abnormality detection sensitivity |
| **Specificity** | **0.8617** | True normal ECG specificity (false alarm resistance) |
| **F1-Score** | **0.7280** | Balanced harmonic mean of precision & recall |
| **ROC-AUC** | **0.8252** | Area Under Receiver Operating Characteristic Curve |

---

## 2. Confusion Matrix Breakdown

| Metric | Value |
|---|---|
| **True Negatives (TN)** | 1184 (Normal correctly classified) |
| **False Positives (FP)** | 190 (Normal incorrectly flagged as abnormal) |
| **False Negatives (FN)** | 703 (Abnormal missed) |
| **True Positives (TP)** | 1195 (Abnormal correctly identified) |

---

## 3. Detailed Classification Report

```text
              precision    recall  f1-score   support

  Normal (0)     0.6275    0.8617    0.7262      1374
Abnormal (1)     0.8628    0.6296    0.7280      1898

    accuracy                         0.7271      3272
   macro avg     0.7451    0.7457    0.7271      3272
weighted avg     0.7640    0.7271    0.7272      3272

```

---

## 4. Methodological Validation & Clinical Integrity
1. **Zero Patient Overlap**: Evaluated strictly on patient IDs never seen during training or validation hyperparameter tuning.
2. **Inductive Graph Inference**: Dynamic k-NN patient similarity graph constructed at test-time without test-train edge leakage.
3. **Multimodal Synergy**: Demonstrates joint representation learning combining tabular demographics (Age, Sex, Height, Weight) with 12-lead potential waveforms.
4. **Visual Artifacts**:
   - Confusion Matrix: `reports/confusion_matrix.png`
   - ROC Curve: `reports/roc_curve.png`
