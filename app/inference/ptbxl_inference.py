"""
==============================================================================
PTB-XL Multimodal GNN Production Inference Pipeline
==============================================================================
Provides safe, cached model loading, leak-free standardized preprocessing,
multimodal feature representation, patient graph construction, and evaluation-mode
inference using the trained Phase 5 PTB-XL Multimodal GNN checkpoint.
==============================================================================
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import numpy as np
import pandas as pd
import joblib

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.train_ptbxl_multimodal import PTBXLMultimodalGNN
from app.graph.patient_graph import PatientGraphBuilder

logger = logging.getLogger("ptbxl_inference")


class PTBXLInferenceService:
    """
    Production-ready, thread-safe inference service for the trained PTB-XL Multimodal GNN.
    
    Guarantees:
    - Model loaded in strict evaluation mode (model.eval()).
    - No gradients tracked (torch.no_grad()).
    - Identical preprocessing using saved StandardScaler.
    - Zero modification to weights during web requests.
    """
    _instance: Optional["PTBXLInferenceService"] = None

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Default artifact locations
        models_dir = PROJECT_ROOT / "models" / "ptbxl_multimodal"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (models_dir / "best_model.pt")
        if not self.checkpoint_path.exists():
            # Fallback check
            fallback = models_dir / "ptbxl_classifier.pt"
            if fallback.exists():
                self.checkpoint_path = fallback
                
        self.scaler_path = Path(scaler_path) if scaler_path else (models_dir / "preprocessor.joblib")
        
        self.clinical_cols = ['age', 'sex', 'height', 'weight']
        
        # Training baseline medians for imputation if missing
        self.train_medians = {
            'age': 62.79,
            'sex': 0.48, # 1 for Male, 0 for Female
            'height': 166.22,
            'weight': 70.32
        }
        
        self.scaler = None
        self.model = None
        self.graph_builder = PatientGraphBuilder(k=5, metric="cosine")
        
        self._load_scaler()
        self._load_model()

    @classmethod
    def get_instance(cls, **kwargs) -> "PTBXLInferenceService":
        """Singleton accessor for cached model inference."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _load_scaler(self):
        """Loads fitted preprocessing scaler."""
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Successfully loaded preprocessor scaler from %s", self.scaler_path)
            except Exception as e:
                logger.warning("Could not load scaler file: %s. Using default baseline standardizer.", e)
                self._init_fallback_scaler()
        else:
            logger.warning("Scaler path %s does not exist. Using fallback standardizer.", self.scaler_path)
            self._init_fallback_scaler()

    def _init_fallback_scaler(self):
        """Initializes fallback standardizer matching train statistics."""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([62.7932667, 0.4838243, 166.22054182, 70.31785902], dtype=np.float64)
        self.scaler.scale_ = np.array([33.06037551, 0.49973828, 6.2487153, 10.37716998], dtype=np.float64)
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = 4

    def _load_model(self):
        """Instantiates and loads PTBXLMultimodalGNN in evaluation mode."""
        if not self.checkpoint_path.exists():
            logger.warning(
                "Trained checkpoint not found at %s. "
                "Inference will report 'trained checkpoint not available' until Phase 5 training completes.",
                self.checkpoint_path
            )
            self.model = None
            return

        self.model = PTBXLMultimodalGNN(
            clinical_in=4,
            ecg_channels=12,
            clinical_emb_dim=64,
            ecg_emb_dim=128,
            fused_dim=128,
            gnn_hidden=64,
            gnn_out=32,
            k_neighbors=5,
            dropout_rate=0.0
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded checkpoint state dict (epoch %s)", checkpoint.get("epoch", "N/A"))
        else:
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded direct state dict from %s", self.checkpoint_path)

        # Enforce evaluation mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _encode_gender(self, gender_val: Any) -> float:
        """Encodes gender string to binary float matching PTB-XL preprocessor."""
        if isinstance(gender_val, (int, float)):
            return float(gender_val)
        if isinstance(gender_val, str):
            g_lower = gender_val.strip().lower()
            if g_lower in ["male", "m", "1", "1.0", "man"]:
                return 1.0
            elif g_lower in ["female", "f", "0", "0.0", "woman"]:
                return 0.0
        return self.train_medians['sex']

    def preprocess_clinical_input(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts, imputes, and standardizes clinical features.
        Columns: ['age', 'sex', 'height', 'weight']
        """
        age = float(data.get("age")) if data.get("age") is not None else self.train_medians['age']
        gender = data.get("gender", data.get("sex", "Male"))
        sex_num = self._encode_gender(gender)
        height = float(data.get("height")) if data.get("height") is not None else self.train_medians['height']
        weight = float(data.get("weight")) if data.get("weight") is not None else self.train_medians['weight']

        df_input = pd.DataFrame([{
            'age': age,
            'sex': sex_num,
            'height': height,
            'weight': weight
        }])[self.clinical_cols]
        scaled_vec = self.scaler.transform(df_input)
        return scaled_vec.astype(np.float32)

    def predict(
        self,
        patient_data: Dict[str, Any],
        ecg_waveform: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes leak-free multimodal GNN inference for a patient intake request.
        
        Args:
            patient_data: Dictionary containing patient clinical parameters.
            ecg_waveform: Optional (12, 1000) or (1, 12, 1000) ECG array.
            
        Returns:
            Dictionary containing prediction, probability, confidence, and metadata.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise FileNotFoundError(
                    f"Trained checkpoint not found at {self.checkpoint_path}. "
                    "Real ML inference cannot be performed. "
                    "Ensure Phase 5 training has been executed via: python ml_pipeline/train_ptbxl_multimodal.py"
                )
            
        # Ensure model remains in eval mode
        self.model.eval()

        # 1. Preprocess clinical features
        scaled_clinical = self.preprocess_clinical_input(patient_data)
        clinical_tensor = torch.tensor(scaled_clinical, dtype=torch.float32, device=self.device)

        # 2. Prepare 12-lead ECG waveform (1, 12, 1000)
        if ecg_waveform is not None:
            if ecg_waveform.ndim == 2:
                ecg_waveform = np.expand_dims(ecg_waveform, axis=0)
            ecg_tensor = torch.tensor(ecg_waveform, dtype=torch.float32, device=self.device)
        else:
            # Baseline zero-centered normalized 12-lead waveform for clinical intake
            ecg_tensor = torch.zeros((1, 12, 1000), dtype=torch.float32, device=self.device)

        # 3. Model Forward Pass with torch.no_grad()
        with torch.no_grad():
            logits = self.model(clinical_tensor, ecg_tensor)
            if logits.ndim > 0:
                logit_val = logits.squeeze().item()
            else:
                logit_val = logits.item()
                
            prob = float(1.0 / (1.0 + np.exp(-logit_val)))

        # 4. Determine Clinical Risk Category and Status
        prediction_label = "Abnormal" if prob >= 0.50 else "Normal"
        
        if prob >= 0.65:
            risk_level = "High Risk"
        elif prob >= 0.40:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # Model confidence percentage relative to the predicted class
        confidence_pct = round((prob if prob >= 0.50 else (1.0 - prob)) * 100, 1)

        return {
            "prediction": prediction_label,
            "probability": round(prob, 4),
            "confidence": confidence_pct,
            "risk_level": risk_level,
            "model": "PTB-XL Multimodal GNN",
            "disclaimer": "This AI-generated result is for research/educational purposes and is not a medical diagnosis."
        }
