"""
=============================================================
PHASE 1 - LOAD AND VERIFY DATA
Healthcare ML Pipeline - NHANES Clinical Data
=============================================================
Purpose:
    Load the available disease-specific cleaned CSV files
    and perform thorough verification of each dataset
    before any modelling work begins.

Available files (located in ML_project/ root):
    - NHANES_diabetes_cleaned.csv       [PRESENT]
    - NHANES_heart_disease_cleaned.csv  [PRESENT]
    - NHANES_CKD_cleaned.csv            [PENDING - not yet available]

NOTED DATASET DIFFERENCES FROM ORIGINAL SPEC:
    - Columns Creatinine, BUN, eGFR are NOT present in the
      diabetes and heart-disease CSVs (kidney biomarkers).
      These columns are expected only in the CKD dataset.
    - SEQN is NOT present in the available CSVs.
      Patient-level tracking will be done via row index.
    - The REQUIRED_FEATURES list is adapted to reflect
      what is actually in each dataset.

Targets:
    Diabetes    (binary 0/1) - in NHANES_diabetes_cleaned.csv
    Heart_Disease (binary 0/1) - in NHANES_heart_disease_cleaned.csv
    CKD         - PENDING
=============================================================
"""

import os
import sys
import io
import pandas as pd
import numpy as np

# Force UTF-8 output (Windows compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Full set of 13 clinical features from the original specification
ALL_REQUIRED_FEATURES = [
    "Age", "Sex", "BMI", "Waist",
    "Systolic_BP", "Diastolic_BP",
    "HDL", "Total_Cholesterol",
    "Glucose", "HbA1c",
    "Creatinine", "BUN", "eGFR"
]

# Kidney biomarkers - expected only in CKD dataset
KIDNEY_FEATURES = {"Creatinine", "BUN", "eGFR"}

# Dataset definitions
# Set 'required_only' to a subset list override, or None to use ALL_REQUIRED_FEATURES
DATASETS = {
    "Diabetes": {
        "file":   "NHANES_diabetes_cleaned.csv",
        "target": "Diabetes",
        "status": "AVAILABLE",
        "notes":  "Kidney biomarkers (Creatinine/BUN/eGFR) not expected in this dataset."
    },
    "Heart Disease": {
        "file":   "NHANES_heart_disease_cleaned.csv",
        "target": "Heart_Disease",
        "status": "AVAILABLE",
        "notes":  "Kidney biomarkers (Creatinine/BUN/eGFR) not expected in this dataset."
    },
    "CKD": {
        "file":   "NHANES_CKD_cleaned.csv",
        "target": "CKD",
        "status": "PENDING",
        "notes":  "File not yet available. Will be added when provided."
    },
}

# Where to look for CSV files (script lives in ml_pipeline/, CSVs are in ML_project/)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIRS = [
    PROJECT_DIR,                             # ML_project/ (primary location)
    os.path.join(SCRIPT_DIR, "data"),        # ml_pipeline/data/
    SCRIPT_DIR,                              # ml_pipeline/
    os.path.join(PROJECT_DIR, "data"),       # ML_project/data/
]

SEQN_COL = "SEQN"

OK   = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def sep(char="=", width=70):
    print(char * width)

def section(title):
    sep()
    print(f"  {title}")
    sep()

def find_file(filename):
    for d in DATA_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None

# ─────────────────────────────────────────────
# PHASE 1 MAIN
# ─────────────────────────────────────────────

def phase1():
    section("PHASE 1 - LOAD AND VERIFY DATA  (Healthcare ML Pipeline)")
    print()
    print(f"  Script location : {SCRIPT_DIR}")
    print(f"  Project root    : {PROJECT_DIR}")
    print()

    problems  = []
    warnings  = []
    summaries = {}

    for disease_name, cfg in DATASETS.items():
        filename = cfg["file"]
        target   = cfg["target"]
        status   = cfg["status"]
        notes    = cfg["notes"]

        sep("-")
        print(f"  Dataset : {disease_name}")
        print(f"  File    : {filename}")
        print(f"  Target  : {target}")
        print(f"  Status  : {status}")
        if notes:
            print(f"  Notes   : {notes}")
        sep("-")
        print()

        # Skip pending datasets
        if status == "PENDING":
            print(f"  {INFO} This dataset is not yet available. Skipping.")
            print(f"         Will be processed once the file is provided.\n")
            warnings.append(f"PENDING: '{filename}' not yet available - CKD phase deferred.")
            continue

        # ── Locate file ───────────────────────────────────────
        path = find_file(filename)
        if path is None:
            msg = (
                f"FILE MISSING: '{filename}' not found.\n"
                + "  Searched:\n"
                + "\n".join(f"    - {d}" for d in DATA_DIRS)
            )
            print(f"  {FAIL} {msg}\n")
            problems.append(msg)
            continue

        print(f"  {OK} Found at: {path}\n")

        # ── Load CSV ──────────────────────────────────────────
        try:
            df = pd.read_csv(path)
        except Exception as e:
            msg = f"LOAD ERROR: '{filename}': {e}"
            print(f"  {FAIL} {msg}\n")
            problems.append(msg)
            continue

        # ── Shape ─────────────────────────────────────────────
        print(f"  Shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")

        # ── Column names ──────────────────────────────────────
        print(f"\n  Columns ({len(df.columns)}):")
        for c in df.columns:
            print(f"    - {c}")

        # ── Data types ────────────────────────────────────────
        print(f"\n  Data types:")
        for col, dtype in df.dtypes.items():
            print(f"    {col:<25} {dtype}")

        # ── Feature verification ──────────────────────────────
        print(f"\n  Clinical feature check (against full 13-feature spec):")
        local_missing = []
        for feat in ALL_REQUIRED_FEATURES:
            present = feat in df.columns
            is_kidney = feat in KIDNEY_FEATURES
            if present:
                print(f"    {OK} {feat}")
            elif is_kidney and disease_name != "CKD":
                # Kidney features are expected absent in non-CKD datasets
                print(f"    {WARN} {feat}  <-- absent (kidney biomarker, expected only in CKD dataset)")
            else:
                print(f"    {FAIL} {feat}  <-- MISSING (unexpected absence)")
                local_missing.append(feat)

        if local_missing:
            msg = f"UNEXPECTED MISSING FEATURES in '{filename}': {local_missing}"
            problems.append(msg)
            print(f"\n  {FAIL} Unexpected missing features: {local_missing}")

        # ── SEQN check ────────────────────────────────────────
        print(f"\n  Patient identifier check (SEQN):")
        if SEQN_COL in df.columns:
            n_uniq = df[SEQN_COL].nunique()
            n_dup  = df.duplicated(subset=[SEQN_COL]).sum()
            print(f"    {OK} SEQN present - {n_uniq:,} unique IDs, {n_dup} duplicates")
        else:
            print(f"    {WARN} SEQN column NOT present.")
            print(f"           Patient tracking will use row index.")
            print(f"           NOTE: The Colab-generated CSVs dropped SEQN during cleaning.")
            warnings.append(f"SEQN absent in '{filename}' - row index will be used for tracking.")

        # ── Target verification ───────────────────────────────
        print(f"\n  Target column ('{target}'):")
        if target not in df.columns:
            msg = f"MISSING TARGET: '{target}' not in '{filename}'."
            print(f"    {FAIL} {msg}")
            problems.append(msg)
            continue

        vc = df[target].value_counts(dropna=False).sort_index()
        total = len(df)
        non_null = df[target].notna().sum()

        for val, cnt in vc.items():
            pct = cnt / total * 100
            print(f"    Value {val!s:<6}: {cnt:>6,}  ({pct:.1f}%)")

        # Binary check
        unique_vals = set(df[target].dropna().unique())
        unexpected_vals = unique_vals - {0, 1, 0.0, 1.0}
        if unexpected_vals:
            msg = (f"UNEXPECTED TARGET VALUES in '{filename}', "
                   f"target='{target}': {unexpected_vals}")
            print(f"\n    {FAIL} {msg}")
            problems.append(msg)
        else:
            pos = int(df[target].sum())
            neg = int((df[target] == 0).sum())
            pos_pct = pos / non_null * 100 if non_null > 0 else 0
            print(f"\n    {OK} Binary target confirmed (0 / 1 only)")
            print(f"    Positive (disease=1) : {pos:>6,}  ({pos_pct:.1f}%)")
            print(f"    Negative (disease=0) : {neg:>6,}  ({100-pos_pct:.1f}%)")
            print(f"    Class imbalance ratio: 1 : {neg/pos:.1f}  (neg:pos)")

        # ── Missing values ────────────────────────────────────
        print(f"\n  Missing values per column:")
        mv = df.isnull().sum()
        any_missing = False
        for col, cnt in mv.items():
            if cnt > 0:
                pct = cnt / total * 100
                print(f"    {col:<25} {cnt:>6,}  ({pct:.1f}%)")
                any_missing = True
        if not any_missing:
            print(f"    {OK} No missing values.")
        else:
            print(f"\n    {INFO} Missing values will be handled via median imputation")
            print(f"           in Phase 4 (fitted on training data ONLY).")

        # ── Duplicate rows ────────────────────────────────────
        n_dup = df.duplicated().sum()
        print(f"\n  Duplicate rows         : {n_dup:,}")
        if n_dup > 0:
            print(f"    {WARN} {n_dup} fully duplicated rows detected.")
        else:
            print(f"    {OK} No duplicate rows.")

        # ── Infinite values ────────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_inf = np.isinf(df[numeric_cols].values).sum()
        print(f"  Infinite values        : {n_inf:,}")
        if n_inf > 0:
            print(f"    {WARN} {n_inf} infinite values found in numeric columns.")
            problems.append(f"INFINITE VALUES: {n_inf} found in '{filename}'.")
        else:
            print(f"    {OK} No infinite values.")

        # ── Summary record ─────────────────────────────────────
        summaries[disease_name] = {
            "rows":          df.shape[0],
            "columns":       df.shape[1],
            "positive":      int(df[target].sum()),
            "negative":      int((df[target] == 0).sum()),
            "pos_pct":       round(pos_pct, 1),
            "missing_cols":  list(mv[mv > 0].index),
            "dup_rows":      int(n_dup),
        }

        print()

    # ─────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────
    sep()
    print("  PHASE 1 SUMMARY TABLE")
    sep()
    print(f"  {'Disease':<15} {'Rows':>7} {'Cols':>5} {'Positive':>9} {'Negative':>9} {'Pos%':>6} {'Dup Rows':>9}")
    sep("-")
    for d, s in summaries.items():
        print(f"  {d:<15} {s['rows']:>7,} {s['columns']:>5} {s['positive']:>9,} "
              f"{s['negative']:>9,} {s['pos_pct']:>5.1f}% {s['dup_rows']:>9}")
    print()

    # ─────────────────────────────────────────
    # PROBLEMS & WARNINGS
    # ─────────────────────────────────────────
    if warnings:
        sep("-")
        print("  WARNINGS (non-blocking):")
        for i, w in enumerate(warnings, 1):
            print(f"  W{i}: {w}")
        print()

    sep()
    if not problems:
        print("  [PHASE 1 COMPLETED SUCCESSFULLY]")
        print()
        print("  Both available datasets loaded and verified.")
        print("  CKD dataset is pending - will be incorporated once provided.")
        print("  Ready to proceed to PHASE 2 (Data Quality Check).")
    else:
        print("  [PHASE 1 COMPLETED WITH PROBLEMS - ACTION REQUIRED]")
        print()
        for i, p in enumerate(problems, 1):
            print(f"  Problem {i}: {p}")
        print()
        print("  *** Resolve these before proceeding to Phase 2. ***")
    sep()

    return problems


if __name__ == "__main__":
    issues = phase1()
    sys.exit(0 if not issues else 1)
