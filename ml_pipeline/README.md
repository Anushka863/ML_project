# ML Pipeline — NHANES Healthcare Prediction

This folder contains the machine-learning pipeline for predicting **Diabetes**, **Heart Disease**, and **CKD** using NHANES clinical data and a Graph Neural Network (GNN).

## Structure

```
ml_pipeline/
├── phase1_load_verify.py     # Phase 1: Load & verify datasets
├── phase1_results.txt        # Phase 1 output log
└── README.md                 # This file
```

*(More phases will be added as the pipeline progresses.)*

## Datasets

| File | Status | Rows | Target |
|------|--------|------|--------|
| `NHANES_diabetes_cleaned.csv` | ✅ Available | 11,452 | `Diabetes` (0/1) |
| `NHANES_heart_disease_cleaned.csv` | ✅ Available | 7,770 | `Heart_Disease` (0/1) |
| `NHANES_CKD_cleaned.csv` | ⏳ Pending | — | `CKD` (0/1) |

## Clinical Features (per dataset)

| Feature | Diabetes | Heart Disease | CKD |
|---------|----------|---------------|-----|
| Age | ✅ | ✅ | ✅ |
| Sex | ✅ | ✅ | ✅ |
| BMI | ✅ | ✅ | ✅ |
| Waist | ✅ | ✅ | ✅ |
| Systolic_BP | ✅ | ✅ | ✅ |
| Diastolic_BP | ✅ | ✅ | ✅ |
| HDL | ✅ | ✅ | ✅ |
| Total_Cholesterol | ✅ | ✅ | ✅ |
| Glucose | ✅ | ✅ | ✅ |
| HbA1c | ✅ | ✅ | ✅ |
| Creatinine | — | — | ✅ |
| BUN | — | — | ✅ |
| eGFR | — | — | ✅ |

## Phase 1 Key Findings

- **No data leakage issues** — no test data seen yet
- **Class imbalance**: ~9-10% positive rate for both available diseases (expected for NHANES population)
- **Missing values**: Significant in lab features (will be handled by median imputation in Phase 4, fitted on train only)
- **SEQN**: Not present in the Colab-generated CSVs — row index will be used for patient tracking
- **Duplicate rows**: 2,873 in diabetes dataset, 1,556 in heart disease dataset (noted for Phase 2 quality check)

## Pipeline Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Load & Verify Data | ✅ Done |
| 2 | Data Quality Check | 🔲 Next |
| 3 | Train/Test Split | 🔲 Pending |
| 4 | Preprocessing (Imputation) | 🔲 Pending |
| 5 | Feature Scaling | 🔲 Pending |
| 6 | Baseline ML (LR + RF) | 🔲 Pending |
| 7 | Patient Graph Construction | 🔲 Pending |
| 8 | GNN Implementation | 🔲 Pending |
| 9 | GNN Training | 🔲 Pending |
| 10 | Final GNN Evaluation | 🔲 Pending |
| 11 | Model Comparison | 🔲 Pending |
| 12 | GNN Explainability | 🔲 Pending |
| 13 | Save ML Pipeline | 🔲 Pending |
| 14 | Backend Integration Prep | 🔲 Pending |

## Running Phase 1

```bash
# From ML_project/ directory:
python -X utf8 ml_pipeline/phase1_load_verify.py
```

Results are saved to `ml_pipeline/phase1_results.txt`.
