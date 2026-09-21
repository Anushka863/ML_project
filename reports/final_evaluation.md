# Final Evaluation Report — PTB-XL Multimodal GNN

**Target**: Binary Heart Disease Detection (Normal vs Abnormal)  
**Evaluation Set**: Untouched Patient Test Set (3272 recordings, 2831 unique patients)  
**Hardware Device**: CPU  
**Evaluation Time**: 26.02s  
**Date**: 2026-09-21 09:03:31  

---

## 1. Quantitative Performance Metrics

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **Accuracy** | **85.27%** | Overall correct classification rate across test cohort |
| **Precision** | **0.8848** | True abnormal proportion among positive alarms |
| **Recall / Sensitivity** | **0.8577** | Cardiac abnormality detection sensitivity |
| **Specificity** | **0.8457** | True normal ECG specificity (false alarm resistance) |
| **F1-Score** | **0.8711** | Balanced harmonic mean of precision & recall |
| **ROC-AUC** | **0.9230** | Area Under Receiver Operating Characteristic Curve |

---

## 2. Confusion Matrix Breakdown

| Metric | Value |
|---|---|
| **True Negatives (TN)** | 1162 (Normal correctly classified) |
| **False Positives (FP)** | 212 (Normal incorrectly flagged as abnormal) |
| **False Negatives (FN)** | 270 (Abnormal missed) |
| **True Positives (TP)** | 1628 (Abnormal correctly identified) |

---

## 3. Detailed Classification Report

```text
              precision    recall  f1-score   support

  Normal (0)     0.8115    0.8457    0.8282      1374
Abnormal (1)     0.8848    0.8577    0.8711      1898

    accuracy                         0.8527      3272
   macro avg     0.8481    0.8517    0.8496      3272
weighted avg     0.8540    0.8527    0.8531      3272

```

---

## 4. Methodological Validation & Clinical Integrity
1. **Zero Patient Overlap**: Evaluated strictly on patient IDs never seen during training or validation hyperparameter tuning.
2. **Inductive Graph Inference**: Dynamic k-NN patient similarity graph constructed at test-time without test-train edge leakage.
3. **Multimodal Synergy**: Demonstrates joint representation learning combining tabular demographics (Age, Sex, Height, Weight) with 12-lead potential waveforms.
4. **Visual Artifacts**:
   - Confusion Matrix: `reports/confusion_matrix.png`
   - ROC Curve: `reports/roc_curve.png`
