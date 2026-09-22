# Multi-Disease GNN Model Training & Test Evaluation Report

**Date:** 2026-09-22  
**Model Checkpoint:** `models/multidisease_gnn_best.pt`  
**Architecture:** `ClinicalEncoder` (12 $\to$ 64) + k-NN Graph Builder ($k=5$) + `MultiDiseaseGNN` (64 $\to$ 64 $\to$ 32) + Multi-Head Logits

## Test Performance Metrics (Strict Held-Out Partition)

| Disease Target | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Diabetes** | **0.9470** | 87.25% | 0.4200 | 0.9018 | 0.5731 |
| **Heart Disease** | **0.7388** | 58.56% | 0.1044 | 0.7547 | 0.1835 |
| **Ckd** | **0.9896** | 94.59% | 0.8780 | 0.9699 | 0.9217 |

- **Total Test Samples:** 1718
- **Leakage Check:** PASS (disjoint patient splits)
