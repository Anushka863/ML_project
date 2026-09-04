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
        
        # 2. Clinical feature handling
        # Impute missing values
        df['age'] = df['age'].fillna(df['age'].median())
        df['height'] = df['height'].fillna(df['height'].median())
        df['weight'] = df['weight'].fillna(df['weight'].median())
        df['sex'] = df['sex'].fillna(0)  # assume 0 mapping for missing sex, or mode
        
        # Normalize
        df[self.clinical_cols] = self.scaler.fit_transform(df[self.clinical_cols])
        
        # Save scaler
        os.makedirs(os.path.join("models", "ptbxl_multimodal"), exist_ok=True)
        joblib.dump(self.scaler, os.path.join("models", "ptbxl_multimodal", "preprocessor.joblib"))

        # 3. Patient level leak-free splitting
        unique_patients = df['patient_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        n_p = len(unique_patients)
        train_p = unique_patients[:int(n_p*0.7)]
        val_p = unique_patients[int(n_p*0.7):int(n_p*0.85)]
        test_p = unique_patients[int(n_p*0.85):]
        
        df_train = df[df['patient_id'].isin(train_p)].copy()
        df_val = df[df['patient_id'].isin(val_p)].copy()
        df_test = df[df['patient_id'].isin(test_p)].copy()
        
        # Report
        os.makedirs("reports", exist_ok=True)
        with open("reports/patient_split_report.md", "w") as f:
            f.write("# Patient Split Report\n\n")
            f.write(f"Total Unique Patients: {n_p}\n")
            f.write(f"- Train Patients: {len(train_p)} (Records: {len(df_train)})\n")
            f.write(f"- Val Patients: {len(val_p)} (Records: {len(df_val)})\n")
            f.write(f"- Test Patients: {len(test_p)} (Records: {len(df_test)})\n\n")
            f.write("Leakage Check:\n")
            check = len(set(train_p).intersection(set(test_p)))
            f.write(f"- Train-Test Intersection: {check}\n")
            
        print("Saving processed metadata splits...")
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
