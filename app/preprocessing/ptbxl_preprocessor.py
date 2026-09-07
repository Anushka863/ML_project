import os
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import StandardScaler
import joblib

class PTBXLPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.db_path = os.path.join(self.data_dir, "ptbxl_database.csv")
        self.scaler = StandardScaler()
        self.clinical_cols = ['age', 'sex', 'height', 'weight']
    
    def _is_normal(self, scp_codes_str):
        try:
            d = ast.literal_eval(scp_codes_str)
            if 'NORM' in d:
                return 0
        except:
            pass
        return 1

    def process_data(self):
        print("Loading PTB-XL database...")
        df = pd.read_csv(self.db_path)
        
        # 1. Target
        df['target'] = df['scp_codes'].apply(self._is_normal)
        
        # 2. Patient level leak-free splitting BEFORE imputation & scaling
        unique_patients = df['patient_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        n_p = len(unique_patients)
        train_p = set(unique_patients[:int(n_p*0.7)])
        val_p = set(unique_patients[int(n_p*0.7):int(n_p*0.85)])
        test_p = set(unique_patients[int(n_p*0.85):])
        
        # Verify 3-way patient disjointness
        train_val_overlap = len(train_p.intersection(val_p))
        train_test_overlap = len(train_p.intersection(test_p))
        val_test_overlap = len(val_p.intersection(test_p))
        assert train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0, "Patient ID overlap detected between splits!"
        
        df_train = df[df['patient_id'].isin(train_p)].copy()
        df_val = df[df['patient_id'].isin(val_p)].copy()
        df_test = df[df['patient_id'].isin(test_p)].copy()
        
        # 3. Clinical feature imputation using TRAIN statistics only
        age_med = df_train['age'].median()
        height_med = df_train['height'].median()
        weight_med = df_train['weight'].median()
        sex_mode = df_train['sex'].mode()[0] if not df_train['sex'].mode().empty else 0
        
        for split_df in [df_train, df_val, df_test]:
            split_df['age'] = split_df['age'].fillna(age_med)
            split_df['height'] = split_df['height'].fillna(height_med)
            split_df['weight'] = split_df['weight'].fillna(weight_med)
            split_df['sex'] = split_df['sex'].fillna(sex_mode)
            
        # 4. Standardize using TRAIN statistics only
        self.scaler.fit(df_train[self.clinical_cols])
        df_train[self.clinical_cols] = self.scaler.transform(df_train[self.clinical_cols])
        df_val[self.clinical_cols] = self.scaler.transform(df_val[self.clinical_cols])
        df_test[self.clinical_cols] = self.scaler.transform(df_test[self.clinical_cols])
        
        # Save scaler and imputation statistics
        os.makedirs(os.path.join("models", "ptbxl_multimodal"), exist_ok=True)
        joblib.dump(self.scaler, os.path.join("models", "ptbxl_multimodal", "preprocessor.joblib"))
        
        # Report
        os.makedirs("reports", exist_ok=True)
        with open("reports/patient_split_report.md", "w") as f:
            f.write("# Patient Split Report\n\n")
            f.write(f"Total Unique Patients: {n_p}\n")
            f.write(f"- Train Patients: {len(train_p)} (Records: {len(df_train)})\n")
            f.write(f"- Val Patients: {len(val_p)} (Records: {len(df_val)})\n")
            f.write(f"- Test Patients: {len(test_p)} (Records: {len(df_test)})\n\n")
            f.write("Leakage Check:\n")
            f.write(f"- Train-Val Intersection: {train_val_overlap}\n")
            f.write(f"- Train-Test Intersection: {train_test_overlap}\n")
            f.write(f"- Val-Test Intersection: {val_test_overlap}\n")
            f.write("- Leakage Status: PASS (0 overlap)\n")
            
        print("Saving processed metadata splits...")
        os.makedirs("data/ptbxl", exist_ok=True)
        df_train.to_csv("data/ptbxl/train_metadata.csv", index=False)
        df_val.to_csv("data/ptbxl/val_metadata.csv", index=False)
        df_test.to_csv("data/ptbxl/test_metadata.csv", index=False)
        
        return df_train, df_val, df_test

    def extract_waveform(self, filename_lr):
        """
        Loads 100Hz waveform using WFDB.
        Ex: filename_lr = 'records100/00000/00001_lr'
        """
        path = os.path.join(self.data_dir, filename_lr)
        try:
            record = wfdb.rdrecord(path)
            signal = record.p_signal # shape: (1000, 12) for 100Hz, 10s
            # Normalize waveform values (z-score per lead)
            mean = np.mean(signal, axis=0, keepdims=True)
            std = np.std(signal, axis=0, keepdims=True) + 1e-8
            signal = (signal - mean) / std
            
            # shape (12, 1000) for PyTorch 1D CNN
            return signal.T
        except Exception as e:
            # return zero array in case of corruption
            return np.zeros((12, 1000))
