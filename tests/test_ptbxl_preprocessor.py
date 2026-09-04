import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This imports the actual file
# BUT if data is not generated yet, testing real files in CI can be hard. We'll test the leak-free requirement.

def test_leakage_prevention():
    base_dir = r"c:\Users\Aisha Fathima\OneDrive\Desktop\ML Project\ML_project"
    data_dir = os.path.join(base_dir, "data", "ptbxl")
    
    train_path = os.path.join(data_dir, "train_metadata.csv")
    test_path = os.path.join(data_dir, "test_metadata.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Skipping leakage check inside CI/test because CSVs do not exist.")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_pts = set(train_df['patient_id'].unique())
    test_pts = set(test_df['patient_id'].unique())
    
    intersection = train_pts.intersection(test_pts)
    assert len(intersection) == 0, f"DATA LEAKAGE DETECTED! {len(intersection)} patients are in both train and test."
    print("test_leakage_prevention passed - 0 patients overlap!")

if __name__ == "__main__":
    test_leakage_prevention()
