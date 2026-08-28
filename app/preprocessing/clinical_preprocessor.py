"""
Clinical Data Preprocessor module.
Phase 3 — Handles loading, feature selection, missing value imputation, scaling,
encoding, and train/val/test splitting without data leakage.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DEFAULT_NUMERICAL_FEATURES = [
    "age", "bmi", "waist", "systolic_bp", "diastolic_bp",
    "hdl", "total_cholesterol", "glucose", "hba1c",
    "creatinine", "bun", "egfr"
]

DEFAULT_CATEGORICAL_FEATURES = ["sex"]

DEFAULT_TARGETS = ["diabetes", "heart_disease", "ckd"]


class ClinicalPreprocessor:
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.numerical_features = numerical_features or DEFAULT_NUMERICAL_FEATURES
        self.categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
        self.target_columns = target_columns or DEFAULT_TARGETS
        self.random_state = random_state
        
        # Preprocessing sub-modules (fitted on train set only)
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.is_fitted = False
        self.fitted_feature_names: List[str] = []

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standardized feature names."""
        df = df.copy()
        column_map = {}
        for col in df.columns:
            clow = col.lower().strip()
            if clow in ["gender", "sex"]:
                column_map[col] = "sex"
            elif clow in ["systolic", "systolic_bp", "sys_bp"]:
                column_map[col] = "systolic_bp"
            elif clow in ["diastolic", "diastolic_bp", "dia_bp"]:
                column_map[col] = "diastolic_bp"
            elif clow in ["cholesterol", "total_cholesterol", "totalcholesterol"]:
                column_map[col] = "total_cholesterol"
            elif clow in ["hba1c", "a1c"]:
                column_map[col] = "hba1c"
            elif clow in ["waist", "waist_circumference"]:
                column_map[col] = "waist"
        return df.rename(columns=column_map)

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        """
        Fit imputer, scaler, and encoder ONLY on training data to prevent data leakage.
        """
        df_norm = self._normalize_column_names(df)
        
        # Select existing numerical features
        num_cols = [c for c in self.numerical_features if c in df_norm.columns]
        cat_cols = [c for c in self.categorical_features if c in df_norm.columns]
        
        if num_cols:
            num_data = df_norm[num_cols].values
            num_imputed = self.num_imputer.fit_transform(num_data)
            self.scaler.fit(num_imputed)
            
        if cat_cols:
            cat_data = df_norm[cat_cols].astype(str).values
            self.cat_encoder.fit(cat_data)
            
        # Record fitted feature order
        fitted_names = list(num_cols)
        if cat_cols and hasattr(self.cat_encoder, "get_feature_names_out"):
            fitted_names.extend(list(self.cat_encoder.get_feature_names_out(cat_cols)))
        self.fitted_feature_names = fitted_names
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new dataframe (or single patient dictionary converted to DataFrame)
        using previously fitted parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted before calling transform().")
            
        df_norm = self._normalize_column_names(df)
        
        num_cols = [c for c in self.numerical_features if c in df_norm.columns]
        cat_cols = [c for c in self.categorical_features if c in df_norm.columns]
        
        parts = []
        if num_cols:
            num_data = df_norm[num_cols].values
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.scaler.transform(num_imputed)
            parts.append(num_scaled)
            
        if cat_cols:
            cat_data = df_norm[cat_cols].astype(str).values
            cat_encoded = self.cat_encoder.transform(cat_data)
            parts.append(cat_encoded)
            
        if not parts:
            raise ValueError("No matching numerical or categorical features found in input DataFrame.")
            
        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed array."""
        return self.fit(df).transform(df)

    def prepare_train_val_test_splits(
        self,
        df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Split dataset into Train, Validation, and Test sets.
        CRITICAL: Fit preprocessing ONLY on the training split to prevent data leakage.
        """
        df_norm = self._normalize_column_names(df)
        
        # Extract targets if present
        available_targets = [t for t in self.target_columns if t in df_norm.columns]
        if available_targets:
            y = df_norm[available_targets].copy()
        else:
            y = pd.DataFrame(index=df_norm.index)
            
        X = df_norm.drop(columns=available_targets, errors="ignore")
        
        # Step 1: Train vs Temp (Val + Test)
        temp_size = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size, random_state=self.random_state
        )
        
        # Step 2: Val vs Test
        val_ratio_in_temp = val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_ratio_in_temp), random_state=self.random_state
        )
        
        # Fit ONLY on X_train to avoid data leakage!
        self.fit(X_train)
        
        X_train_proc = self.transform(X_train)
        X_val_proc = self.transform(X_val)
        X_test_proc = self.transform(X_test)
        
        return {
            "X_train": X_train_proc,
            "y_train": y_train,
            "X_val": X_val_proc,
            "y_val": y_val,
            "X_test": X_test_proc,
            "y_test": y_test,
            "feature_names": self.fitted_feature_names,
            "target_names": available_targets
        }

    def save(self, file_path: str):
        """Save fitted preprocessor state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "ClinicalPreprocessor":
        """Load fitted preprocessor from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at: {file_path}")
        return joblib.load(file_path)
