"""
ODIR-5K Ophthalmic Data Preprocessor module.
Dedicated preprocessing for ODIR-5K dataset:
- Standardizes patient demographics (Age and Sex).
- Validates and transforms bilateral retinal fundus images (Left and Right eyes).
- Multi-label target vector encoding (N, D, G, C, A, H, M, O).
- Isolated from NHANES and PTB-XL preprocessors.
"""
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
from PIL import Image
import joblib

from app.preprocessing.image_preprocessor import ImagePreprocessor

ODIR_DISEASE_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_DISEASE_NAMES = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other"
}


class ODIRPreprocessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None
    ):
        self.image_size = image_size
        self.age_mean = age_mean if age_mean is not None else 57.854
        self.age_std = age_std if age_std is not None else 11.724
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.is_fitted = age_mean is not None and age_std is not None

    def fit(self, df: pd.DataFrame) -> "ODIRPreprocessor":
        """
        Fit demographic parameters on training dataframe.
        """
        age_series = pd.to_numeric(df["Patient Age"], errors="coerce").dropna()
        self.age_mean = float(age_series.mean())
        self.age_std = float(age_series.std()) if age_series.std() > 0 else 1.0
        self.is_fitted = True
        return self

    def preprocess_demographics(self, age: float, sex: Union[str, int]) -> torch.Tensor:
        """
        Preprocess age and sex into normalized 2D feature tensor.
        sex: 'Male' / 1 -> 1.0, 'Female' / 0 -> 0.0
        Returns Tensor of shape (1, 2).
        """
        # Encode sex
        if isinstance(sex, str):
            sex_val = 1.0 if sex.strip().lower() in ["male", "m", "1"] else 0.0
        else:
            sex_val = 1.0 if sex == 1 else 0.0

        # Normalize age
        norm_age = (float(age) - self.age_mean) / (self.age_std + 1e-8)
        
        feature_vec = torch.tensor([[norm_age, sex_val]], dtype=torch.float32)
        return feature_vec

    def preprocess_bilateral_images(
        self,
        left_img: Union[str, Path, Image.Image],
        right_img: Union[str, Path, Image.Image],
        is_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and preprocess both Left and Right eye fundus images.
        Returns:
            left_tensor: (1, 3, 224, 224)
            right_tensor: (1, 3, 224, 224)
        """
        # Process Left Image
        if isinstance(left_img, Image.Image):
            left_tensor = self.image_preprocessor.preprocess_pil_image(left_img, is_training=is_training)
        else:
            left_tensor = self.image_preprocessor.load_and_preprocess(left_img, is_training=is_training)

        # Process Right Image
        if isinstance(right_img, Image.Image):
            right_tensor = self.image_preprocessor.preprocess_pil_image(right_img, is_training=is_training)
        else:
            right_tensor = self.image_preprocessor.load_and_preprocess(right_img, is_training=is_training)

        return left_tensor, right_tensor

    def encode_labels(self, row: Union[pd.Series, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract multi-label ground truth vector (8-dim) for N, D, G, C, A, H, M, O.
        """
        labels = [float(row.get(col, 0)) for col in ODIR_DISEASE_LABELS]
        return torch.tensor(labels, dtype=torch.float32)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor state to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "image_size": self.image_size,
                "age_mean": self.age_mean,
                "age_std": self.age_std,
                "is_fitted": self.is_fitted
            },
            path
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ODIRPreprocessor":
        """Load preprocessor state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ODIR preprocessor artifact not found at: {path}")
        data = joblib.load(path)
        preprocessor = cls(
            image_size=data["image_size"],
            age_mean=data["age_mean"],
            age_std=data["age_std"]
        )
        preprocessor.is_fitted = data.get("is_fitted", True)
        return preprocessor
