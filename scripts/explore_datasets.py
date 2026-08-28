"""
Dataset Exploration & Audit Script
Phase 1 — Step 1: Explore clinical tabular and medical image datasets.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image
except ImportError:
    Image = None

BASE_DIR = Path(__file__).resolve().parent.parent

# Search paths for clinical tabular datasets
CLINICAL_SEARCH_PATHS = [
    BASE_DIR / "data" / "processed" / "NHANES_diabetes_final.csv",
    BASE_DIR / "data" / "processed" / "NHANES_heart_disease_final.csv",
    BASE_DIR / "data" / "processed" / "NHANES_CKD_final.csv",
    BASE_DIR / "data" / "processed" / "NHANES_multidisease_master_final.csv",
    BASE_DIR / "data" / "clinical" / "NHANES" / "NHANES_multidisease_master_final.csv",
]

# Search paths for image datasets
IMAGE_SEARCH_PATHS = {
    "Diabetes (Retinal Fundus)": [
        BASE_DIR / "data" / "raw" / "diabetes_images",
        BASE_DIR / "data" / "imaging" / "diabetes",
    ],
    "CKD (Kidney CT Scans)": [
        BASE_DIR / "data" / "raw" / "ckd_images",
        BASE_DIR / "data" / "imaging" / "ckd",
    ],
    "Heart (PTB-XL ECG)": [
        BASE_DIR / "data" / "raw" / "heart_images",
        BASE_DIR / "data" / "imaging" / "heart",
    ],
}

PATIENT_ID_CANDIDATES = ["seqn", "patient_id", "patientid", "id", "subject_id", "subjectid"]
TARGET_CANDIDATES = ["diabetes", "heart_disease", "ckd", "target", "outcome", "label", "status"]


def audit_clinical_datasets():
    print("=" * 70)
    print("📋 CLINICAL TABULAR DATASETS AUDIT")
    print("=" * 70)
    
    found_count = 0
    clinical_reports = []
    
    for path in CLINICAL_SEARCH_PATHS:
        if not path.exists():
            continue
            
        found_count += 1
        report = {"filename": path.name, "path": str(path)}
        print(f"\n✅ Found Clinical Dataset: {path.name}")
        
        if pd is not None:
            try:
                df = pd.read_csv(path)
                report["num_rows"] = len(df)
                report["num_cols"] = len(df.columns)
                
                # Identify Patient ID column
                pid_col = next((c for c in df.columns if c.lower() in PATIENT_ID_CANDIDATES), None)
                report["patient_id_col"] = pid_col
                
                # Identify Targets
                target_cols = [c for c in df.columns if any(t in c.lower() for t in TARGET_CANDIDATES) and c.lower() not in PATIENT_ID_CANDIDATES]
                report["target_cols"] = target_cols
                
                # Features are non-id, non-target columns
                feature_cols = [c for c in df.columns if c != pid_col and c not in target_cols]
                report["feature_cols"] = feature_cols
                
                # Missing values count
                missing_total = int(df.isnull().sum().sum())
                report["missing_count"] = missing_total
                
                print(f"   • File: {path.name}")
                print(f"   • Rows: {len(df)} | Columns: {len(df.columns)}")
                print(f"   • Patient ID Column: {pid_col if pid_col else 'NOT DETECTED'}")
                print(f"   • Clinical Features ({len(feature_cols)}): {feature_cols[:8]}{'...' if len(feature_cols)>8 else ''}")
                print(f"   • Target Columns: {target_cols}")
                print(f"   • Total Missing Values: {missing_total}")
                
                if target_cols:
                    print("   • Class Distributions:")
                    for tc in target_cols:
                        dist = df[tc].value_counts(dropna=False).to_dict()
                        print(f"     - {tc}: {dist}")
                        
                clinical_reports.append(report)
            except Exception as e:
                print(f"   ⚠️ Error loading CSV: {e}")
        else:
            print("   ⚠️ pandas not installed.")
            
    if found_count == 0:
        print("\n❌ No clinical tabular CSV files found on disk.")
        print("   Expected paths included:")
        for p in CLINICAL_SEARCH_PATHS:
            print(f"   - {p}")
            
    return clinical_reports


def audit_image_datasets():
    print("\n" + "=" * 70)
    print("🖼️ MEDICAL IMAGE DATASETS AUDIT")
    print("=" * 70)
    
    image_reports = []
    found_any = False
    
    for category, paths in IMAGE_SEARCH_PATHS.items():
        existing_path = next((p for p in paths if p.exists()), None)
        
        if existing_path is None:
            print(f"\n❌ Missing Image Dataset: {category}")
            print(f"   Expected at: {paths[0]}")
            continue
            
        found_any = True
        print(f"\n✅ Found Image Directory: {category} ({existing_path.name})")
        
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}
        image_files = [f for f in existing_path.rglob("*") if f.suffix.lower() in extensions]
        
        report = {
            "category": category,
            "path": str(existing_path),
            "num_images": len(image_files),
            "formats": list(set(f.suffix.lower() for f in image_files)),
        }
        
        print(f"   • Total Images: {len(image_files)}")
        print(f"   • Formats Found: {report['formats']}")
        
        # Check image dimensions sample if PIL available
        if Image is not None and image_files:
            try:
                with Image.open(image_files[0]) as img:
                    report["sample_dimensions"] = img.size
                    report["sample_mode"] = img.mode
                print(f"   • Sample Image Dimensions: {report['sample_dimensions']} (Mode: {report['sample_mode']})")
            except Exception as e:
                print(f"   ⚠️ Could not read sample image: {e}")
                
        # Class distribution via subfolders or CSV labels
        subdirs = [d for d in existing_path.iterdir() if d.is_dir()]
        csv_files = list(existing_path.glob("*.csv"))
        
        if subdirs:
            print("   • Subdirectory Class Breakdown:")
            class_counts = {}
            for sd in subdirs:
                c_count = len([f for f in sd.rglob("*") if f.suffix.lower() in extensions])
                class_counts[sd.name] = c_count
                print(f"     - {sd.name}: {c_count} images")
            report["class_distribution"] = class_counts
            
        if csv_files and pd is not None:
            for cf in csv_files:
                try:
                    df_img = pd.read_csv(cf)
                    print(f"   • Metadata CSV ({cf.name}): {len(df_img)} records")
                    pid_col = next((c for c in df_img.columns if c.lower() in PATIENT_ID_CANDIDATES), None)
                    if pid_col:
                        print(f"     - Patient ID Column in CSV: {pid_col}")
                        print(f"     - Unique Patients: {df_img[pid_col].nunique()}")
                except Exception:
                    pass
                    
        image_reports.append(report)
        
    return image_reports


def main():
    print(f"🔍 Starting Dataset Audit in: {BASE_DIR}\n")
    clinical_reports = audit_clinical_datasets()
    image_reports = audit_image_datasets()
    print("\n" + "=" * 70)
    print("Step 1 Dataset Audit completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
