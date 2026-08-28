"""
Test suite for ClinicalPreprocessor (Phase 3 verification)
"""
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.preprocessing.clinical_preprocessor import ClinicalPreprocessor


def test_clinical_preprocessor():
    print("=" * 70)
    print("🧪 TESTING CLINICAL PREPROCESSOR (PHASE 3 VERIFICATION)")
    print("=" * 70)
    
    # Generate synthetic NHANES-style data with missing values
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "age": np.random.randint(20, 80, size=n_samples).astype(float),
        "sex": np.random.choice(["Male", "Female"], size=n_samples),
        "bmi": np.random.uniform(18.5, 40.0, size=n_samples),
        "waist": np.random.uniform(70.0, 120.0, size=n_samples),
        "systolic_bp": np.random.uniform(100, 180, size=n_samples),
        "diastolic_bp": np.random.uniform(60, 110, size=n_samples),
        "hdl": np.random.uniform(30, 80, size=n_samples),
        "total_cholesterol": np.random.uniform(150, 300, size=n_samples),
        "glucose": np.random.uniform(70, 200, size=n_samples),
        "hba1c": np.random.uniform(4.5, 10.0, size=n_samples),
        "creatinine": np.random.uniform(0.6, 2.5, size=n_samples),
        "bun": np.random.uniform(7, 30, size=n_samples),
        "egfr": np.random.uniform(30, 120, size=n_samples),
        "diabetes": np.random.choice([0, 1], size=n_samples),
        "heart_disease": np.random.choice([0, 1], size=n_samples),
        "ckd": np.random.choice([0, 1], size=n_samples),
    })
    
    # Introduce missing values in 10% of numerical cells
    mask = np.random.rand(*df[["bmi", "glucose", "creatinine"]].shape) < 0.1
    df.loc[mask[:, 0], "bmi"] = np.nan
    df.loc[mask[:, 1], "glucose"] = np.nan
    df.loc[mask[:, 2], "creatinine"] = np.nan

    print(f"• Synthetic Dataset Created: {len(df)} rows, {df.isnull().sum().sum()} missing values")
    
    preprocessor = ClinicalPreprocessor(random_state=42)
    splits = preprocessor.prepare_train_val_test_splits(df, val_size=0.15, test_size=0.15)
    
    print("\n✅ Train/Val/Test Split Completed:")
    print(f"   - X_train shape: {splits['X_train'].shape}")
    print(f"   - X_val shape:   {splits['X_val'].shape}")
    print(f"   - X_test shape:  {splits['X_test'].shape}")
    print(f"   - Features ({len(splits['feature_names'])}): {splits['feature_names']}")
    
    # Assertions
    assert not np.isnan(splits["X_train"]).any(), "Imputation failed: NaN values found in X_train!"
    assert not np.isnan(splits["X_val"]).any(), "Imputation failed: NaN values found in X_val!"
    assert not np.isnan(splits["X_test"]).any(), "Imputation failed: NaN values found in X_test!"
    
    # Test Save & Load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "preprocessor.joblib"
        preprocessor.save(str(save_path))
        print(f"\n✅ Preprocessor saved to: {save_path}")
        
        loaded_prep = ClinicalPreprocessor.load(str(save_path))
        print("✅ Preprocessor loaded successfully.")
        
        # Test inference transform on single patient dictionary
        sample_patient = pd.DataFrame([{
            "age": 55, "sex": "Male", "bmi": 28.4, "waist": 95,
            "systolic_bp": 135, "diastolic_bp": 85, "hdl": 45,
            "total_cholesterol": 210, "glucose": 110, "hba1c": 6.2,
            "creatinine": 1.1, "bun": 16, "egfr": 80
        }])
        
        patient_vector = loaded_prep.transform(sample_patient)
        print(f"✅ Single patient inference vector transformed: shape {patient_vector.shape}")
        assert patient_vector.shape[1] == splits["X_train"].shape[1]
        
    print("\n🎉 ALL CLINICAL PREPROCESSOR TESTS PASSED!")


if __name__ == "__main__":
    test_clinical_preprocessor()
