"""
Patient-Level Linkage Audit Script
Phase 1 — Step 2: Audit patient-level linkage between clinical and image datasets.
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

BASE_DIR = Path(__file__).resolve().parent.parent

CLINICAL_PATHS = [
    BASE_DIR / "data" / "processed" / "NHANES_multidisease_master_final.csv",
    BASE_DIR / "data" / "processed" / "NHANES_diabetes_final.csv",
    BASE_DIR / "data" / "clinical" / "NHANES" / "NHANES_multidisease_master_final.csv",
]

IMAGE_METADATA_PATHS = [
    BASE_DIR / "data" / "raw" / "diabetes_images" / "train.csv",
    BASE_DIR / "data" / "imaging" / "diabetes" / "labels.csv",
    BASE_DIR / "data" / "raw" / "ckd_images" / "metadata.csv",
]

PATIENT_ID_CANDIDATES = ["seqn", "patient_id", "patientid", "id", "subject_id", "subjectid"]


def audit_linkage():
    print("=" * 70)
    print("🔗 PATIENT-LEVEL LINKAGE AUDIT")
    print("=" * 70)
    
    clinical_ids = set()
    image_ids = set()
    
    # 1. Load Clinical Patient IDs
    found_clinical = False
    if pd is not None:
        for cp in CLINICAL_PATHS:
            if cp.exists():
                try:
                    df_c = pd.read_csv(cp)
                    pid_col = next((c for c in df_c.columns if c.lower() in PATIENT_ID_CANDIDATES), None)
                    if pid_col:
                        clinical_ids.update(df_c[pid_col].dropna().astype(str).unique())
                        found_clinical = True
                        print(f"\n✅ Clinical Dataset Found ({cp.name}):")
                        print(f"   • Patient ID Column: '{pid_col}'")
                        print(f"   • Total Unique Clinical Patient IDs: {len(clinical_ids)}")
                        break
                except Exception as e:
                    print(f"   ⚠️ Could not read clinical file {cp.name}: {e}")
                    
    if not found_clinical:
        print("\n❌ No clinical patient IDs found (clinical CSV file missing or no patient ID column).")

    # 2. Load Image Patient IDs
    found_image = False
    if pd is not None:
        for imp in IMAGE_METADATA_PATHS:
            if imp.exists():
                try:
                    df_i = pd.read_csv(imp)
                    pid_col = next((c for c in df_i.columns if c.lower() in PATIENT_ID_CANDIDATES), None)
                    if pid_col:
                        image_ids.update(df_i[pid_col].dropna().astype(str).unique())
                        found_image = True
                        print(f"\n✅ Image Metadata CSV Found ({imp.name}):")
                        print(f"   • Image Patient ID Column: '{pid_col}'")
                        print(f"   • Total Unique Image Patient IDs: {len(image_ids)}")
                        break
                except Exception as e:
                    print(f"   ⚠️ Could not read image metadata file {imp.name}: {e}")

    if not found_image:
        print("\n❌ No image patient IDs found (no image metadata CSV with patient IDs detected).")

    # 3. Compute Linkage Metrics
    matching_ids = clinical_ids.intersection(image_ids)
    clinical_only = clinical_ids - image_ids
    image_only = image_ids - clinical_ids

    linkage_valid = len(matching_ids) > 0

    print("\n" + "-" * 70)
    print("📊 PATIENT LINKAGE METRICS SUMMARY")
    print("-" * 70)
    print(f"• Unique Clinical Patient IDs:    {len(clinical_ids)}")
    print(f"• Unique Image Patient IDs:       {len(image_ids)}")
    print(f"• Matching Cross-Modality IDs:    {len(matching_ids)}")
    print(f"• Clinical-Only Patient IDs:      {len(clinical_only)}")
    print(f"• Image-Only Patient IDs:         {len(image_only)}")
    print(f"• Patient Linkage Status:         {'VALID ✅' if linkage_valid else 'INVALID ❌'}")
    print("-" * 70)

    if not linkage_valid:
        print("\n🚨 CONCLUSION & CRITICAL RULE ENFORCEMENT:")
        print("There is NO documented patient-level linkage (no shared patient IDs) between the NHANES clinical survey and the image datasets.")
        print("CRITICAL RULE ENFORCED:")
        print("1. Records will NOT be merged.")
        print("2. Clinical records and images will NOT be randomly paired.")
        print("3. Representation will NOT be claimed to belong to the same patient.")
        print("4. Status: Multimodal patient-level fusion is NOT READY due to unlinked datasets.")

    return {
        "num_clinical_patients": len(clinical_ids),
        "num_image_patients": len(image_ids),
        "num_matching_patients": len(matching_ids),
        "clinical_only": len(clinical_only),
        "image_only": len(image_only),
        "linkage_valid": linkage_valid,
    }


if __name__ == "__main__":
    audit_linkage()
