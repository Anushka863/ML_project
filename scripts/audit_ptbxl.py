import os
import zipfile
import pandas as pd
import io

ZIP_PATH = r"c:\Users\Aisha Fathima\OneDrive\Desktop\ML Project\ptb-xl-1.0.3.zip"

def audit_ptbxl():
    print("==================================================")
    print("PHASE 1 — PTB-XL DATASET AUDIT")
    print("==================================================")
    
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Could not find {ZIP_PATH}")
        return

    print(f"1. Dataset path: {ZIP_PATH}")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_files = z.namelist()
        
        # Find exactly the right CSVs
        db_csv_path = [f for f in all_files if f.endswith('ptbxl_database.csv')]
        scp_csv_path = [f for f in all_files if f.endswith('scp_statements.csv')]
        
        if not db_csv_path:
            print("Could not find ptbxl_database.csv inside the ZIP.")
            return
        
        print(f"Found database CSV: {db_csv_path[0]}")
        
        with z.open(db_csv_path[0]) as f:
            df = pd.read_csv(f)
            
        print(f"\n2. Number of ECG records: {len(df)}")
        
        if 'patient_id' in df.columns:
            unique_patients = df['patient_id'].nunique()
            print(f"3. Number of unique patients: {unique_patients}")
            print("4. Available patient-level identifiers: 'patient_id'")
            print("5. Whether patient_id exists: TRUE")
        else:
            print("3. Number of unique patients: UNKNOWN")
            print("4. Available patient-level identifiers: NONE")
            print("5. Whether patient_id exists: FALSE")
            
        clinical_cols = ['age', 'sex', 'height', 'weight', 'nurse', 'site', 'device', 'recording_date', 'report']
        avail_clin_cols = [c for c in clinical_cols if c in df.columns]
        print(f"6. Clinical/demographic columns found: {avail_clin_cols}")
        
        print("\n7. Missing values in key clinical columns:")
        for c in avail_clin_cols:
            missing = df[c].isna().sum()
            print(f"   - {c}: {missing} missing ({missing/len(df)*100:.1f}%)")
            
        # File formats and ECG metadata
        print("\n8. ECG file format:")
        has_dat = any(f.endswith('.dat') for f in all_files)
        has_hea = any(f.endswith('.hea') for f in all_files)
        print(f"   WFDB format (.dat and .hea exists): {has_dat and has_hea}")
        
        print(f"9. ECG sampling frequency: Described in columns (100 Hz / 500 Hz representations exist - e.g. filename_lr, filename_hr in dataset)")
        print("10. ECG duration: 10 seconds (standard PTB-XL)")
        print("11. Number of ECG leads: 12 leads (standard PTB-XL)")
        
        print("\n12. Diagnostic labels overview:")
        if 'scp_codes' in df.columns:
            print("   'scp_codes' column found as dictionary string")
            
            # Simple eval to check frequencies
            # Because scp_codes looks like {'NORM': 100.0, 'LMI': 0.0}, we can parse standard labels for a subset
            try:
                import ast
                def extract_primary(x):
                    try:
                        d = ast.literal_eval(x)
                        if isinstance(d, dict) and len(d) > 0:
                            return list(d.keys())[0]
                    except:
                        pass
                    return 'UNKNOWN'
                
                primary_labels = df['scp_codes'].apply(extract_primary)
                counts = primary_labels.value_counts().head(10)
                print("\n13. Label frequencies (Top 10 primary SCP codes):")
                for k, v in counts.items():
                    print(f"    - {k}: {v}")
            except Exception as e:
                print(f"Could not parse scp_codes: {e}")
        
        if 'patient_id' in df.columns:
            records_per_patient = df.groupby('patient_id').size()
            print(f"\n14. Number of records per patient: Min: {records_per_patient.min()}, Max: {records_per_patient.max()}, Mean: {records_per_patient.mean():.2f}")
            print(f"15. Whether multiple ECG records belong to the same patient: {'YES' if records_per_patient.max() > 1 else 'NO'}")
            print("16. How records map to patient_id: Direct mapping via 'patient_id' column where multiple rows can have the same patient_id representing temporal recordings.")
            
        print("\n--- PLAIN ENGLISH SUMMARY ---")
        print("PATIENT:")
        if 'patient_id' in df.columns:
            sample_pt = df.iloc[0]
            print(f"patient_id = {sample_pt['patient_id']}")
            print(f"clinical information = Age: {sample_pt.get('age', 'N/A')}, Sex: {sample_pt.get('sex', 'N/A')}, Weight: {sample_pt.get('weight', 'N/A')}, Height: {sample_pt.get('height', 'N/A')}")
            print(f"ECG records = Located at {sample_pt.get('filename_lr', 'N/A')} and {sample_pt.get('filename_hr', 'N/A')} (WFDB format)")
            print(f"Labels = {sample_pt.get('scp_codes', 'N/A')}")
        else:
            print("No summary available - patient_id missing.")
        print("==================================================")

if __name__ == "__main__":
    audit_ptbxl()
