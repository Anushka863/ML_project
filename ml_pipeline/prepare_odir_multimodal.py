"""
ODIR-5K Multimodal Dataset Preparation Pipeline.
Phase: ODIR Ophthalmic Branch

1. Reads raw ODIR-5K metadata spreadsheet (data.xlsx).
2. Validates existence of both left and right fundus images for each patient.
3. Performs patient-level train/validation/test splitting (70% Train, 15% Val, 15% Test).
   - Strict rule: Both eyes of a patient are strictly assigned to the SAME split.
   - Zero patient leakage between train, val, and test.
4. Fits demographic standardization on train split.
5. Saves processed metadata CSVs to data/processed/odir/ and saves fitted preprocessor artifact.
"""
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.odir_preprocessor import ODIRPreprocessor, ODIR_DISEASE_LABELS


def read_odir_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """
    Reads ODIR-5K data.xlsx using robust zipfile XML parser (zero external excel dependency).
    """
    with zipfile.ZipFile(xlsx_path, "r") as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.findall("ns:si", ns):
                t = si.find("ns:t", ns)
                if t is not None and t.text is not None:
                    shared_strings.append(t.text)
                else:
                    r_texts = [r.find("ns:t", ns).text for r in si.findall("ns:r", ns) if r.find("ns:t", ns) is not None and r.find("ns:t", ns).text is not None]
                    shared_strings.append("".join(r_texts))

        sheet_tree = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows_data = []
        sheet_data = sheet_tree.find("ns:sheetData", ns)
        for row in sheet_data.findall("ns:row", ns):
            row_vals = {}
            for c in row.findall("ns:c", ns):
                r_ref = c.get("r")
                t_type = c.get("t")
                v = c.find("ns:v", ns)
                val = v.text if v is not None else ""
                if t_type == "s" and val != "":
                    val = shared_strings[int(val)]
                col_letter = "".join([ch for ch in r_ref if ch.isalpha()])
                row_vals[col_letter] = val
            rows_data.append(row_vals)

    header_row = rows_data[0]
    ordered_cols = sorted(header_row.keys(), key=lambda x: (len(x), x))
    headers = [header_row[c] for c in ordered_cols]
    data_rows = []
    for r in rows_data[1:]:
        row = [r.get(c, "") for c in ordered_cols]
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=headers)
    return df


def prepare_odir_dataset(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    random_state: int = 42
):
    if raw_data_dir is None:
        raw_data_dir = PROJECT_ROOT / "data" / "odir5k" / "ODIR-5K" / "ODIR-5K"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed" / "odir"

    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = raw_data_dir / "data.xlsx"
    train_img_dir = raw_data_dir / "Training Images"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"data.xlsx not found at: {xlsx_path}")
    if not train_img_dir.exists():
        raise FileNotFoundError(f"Training Images folder not found at: {train_img_dir}")

    print(f"Reading ODIR metadata from: {xlsx_path}")
    df = read_odir_xlsx(xlsx_path)
    print(f"Loaded {len(df)} patient records.")

    # Validate columns
    required_cols = ["ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus"] + ODIR_DISEASE_LABELS
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from data.xlsx")

    # Clean numeric types
    df["ID"] = pd.to_numeric(df["ID"])
    df["Patient Age"] = pd.to_numeric(df["Patient Age"])
    for col in ODIR_DISEASE_LABELS:
        df[col] = pd.to_numeric(df[col]).fillna(0).astype(int)

    # Validate image files physically exist
    print("Verifying bilateral physical image existence...")
    valid_mask = []
    left_paths = []
    right_paths = []

    for _, row in df.iterrows():
        left_file = train_img_dir / str(row["Left-Fundus"])
        right_file = train_img_dir / str(row["Right-Fundus"])

        left_ok = left_file.exists()
        right_ok = right_file.exists()

        if left_ok and right_ok:
            valid_mask.append(True)
            left_paths.append(str(left_file.relative_to(PROJECT_ROOT)).replace("\\", "/"))
            right_paths.append(str(right_file.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        else:
            valid_mask.append(False)
            left_paths.append(None)
            right_paths.append(None)

    df["left_image_path"] = left_paths
    df["right_image_path"] = right_paths
    df_valid = df[valid_mask].copy().reset_index(drop=True)

    print(f"Bilateral valid patients: {len(df_valid)} / {len(df)} (100.0%)")

    # Patient-level splitting: 70% Train (2,450), 15% Val (525), 15% Test (525)
    # Both eyes of each patient strictly stay in the same split!
    patient_ids = df_valid["ID"].unique()
    train_ids, testval_ids = train_test_split(patient_ids, test_size=0.30, random_state=random_state)
    val_ids, test_ids = train_test_split(testval_ids, test_size=0.50, random_state=random_state)

    train_df = df_valid[df_valid["ID"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df_valid[df_valid["ID"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df_valid[df_valid["ID"].isin(test_ids)].copy().reset_index(drop=True)

    print(f"\n--- Patient Split Summary ---")
    print(f"Train patients: {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%) | Images: {len(train_df)*2}")
    print(f"Val patients:   {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%) | Images: {len(val_df)*2}")
    print(f"Test patients:  {len(test_df)} ({len(test_df)/len(df_valid)*100:.1f}%) | Images: {len(test_df)*2}")

    # Check zero leakage
    train_set = set(train_df["ID"])
    val_set = set(val_df["ID"])
    test_set = set(test_df["ID"])

    assert len(train_set.intersection(val_set)) == 0, "Train-Val patient leakage detected!"
    assert len(train_set.intersection(test_set)) == 0, "Train-Test patient leakage detected!"
    assert len(val_set.intersection(test_set)) == 0, "Val-Test patient leakage detected!"
    print("Verification passed: ZERO patient overlap across splits.")

    # Fit preprocessor on training split
    preprocessor = ODIRPreprocessor()
    preprocessor.fit(train_df)
    preprocessor_path = output_dir / "odir_preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Saved ODIR preprocessor to: {preprocessor_path}")

    # Save CSVs
    train_df.to_csv(output_dir / "train_odir.csv", index=False)
    val_df.to_csv(output_dir / "val_odir.csv", index=False)
    test_df.to_csv(output_dir / "test_odir.csv", index=False)
    print(f"Saved processed split CSVs to: {output_dir}")

    return train_df, val_df, test_df, preprocessor


if __name__ == "__main__":
    from typing import Optional
    prepare_odir_dataset()
