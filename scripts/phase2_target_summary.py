import pandas as pd
import ast
import os

def create_phase2_reports():
    base_dir = r"c:\Users\Aisha Fathima\OneDrive\Desktop\ML Project\ML_project"
    data_dir = os.path.join(base_dir, "data", "ptbxl", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    
    if not os.path.exists(db_path):
        print(f"Could not find {db_path}")
        return

    df = pd.read_csv(db_path)

    # Determine binary target: NORM vs Abnormal
    # Using 100% confidence approach: if NORM is the primary/absolute diagnostic label
    def is_normal(scp_codes_str):
        try:
            d = ast.literal_eval(scp_codes_str)
            if 'NORM' in d: # Some papers say if NORM > 0.0 or just 'NORM' is in the dict
                # PTB-XL assigns probability. Usually 'NORM': 100.0
                return 0
        except:
            pass
        return 1

    df['target'] = df['scp_codes'].apply(is_normal)
    
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Label distribution
    counts = df['target'].value_counts().rename(index={0: 'Normal', 1: 'Abnormal'})
    counts.to_csv(os.path.join(reports_dir, "ptbxl_label_distribution.csv"), index_label='Class', header=['Count'])
    
    num_patients = df['patient_id'].nunique()
    num_records = len(df)
    norm_count = counts.get('Normal', 0)
    abnorm_count = counts.get('Abnormal', 0)
    
    summary_md = f"""# PTB-XL Target Summary

## TARGET DEFINITION
**Positive class (1)**: Abnormal ECG (all other diagnostic classes)
**Negative class (0)**: Normal ECG (NORM)
**Reason**: To create a robust, binary baseline proof-of-concept for detecting cardiac abnormalities before mapping granular classes to complex NHANES equivalents.
**Source label(s)**: `scp_codes` dict containing 'NORM'.

## STATISTICS
* **Number of patients**: {num_patients}
* **Number of ECG records**: {num_records}
* **Normal count (Negative)**: {norm_count}
* **Abnormal count (Positive)**: {abnorm_count}

## CLASS BALANCE
* **Normal %**: {(norm_count/num_records)*100:.2f}%
* **Abnormal %**: {(abnorm_count/num_records)*100:.2f}%
"""

    with open(os.path.join(reports_dir, "ptbxl_target_summary.md"), "w") as f:
        f.write(summary_md)
        
    print("Phase 2 reports generated.")

if __name__ == "__main__":
    create_phase2_reports()
