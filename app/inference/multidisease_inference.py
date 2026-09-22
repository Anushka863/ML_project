"""
==============================================================================
Multi-Disease GNN Production Inference Pipeline
==============================================================================
Provides safe, cached model loading, standardized preprocessing, dynamic
graph representation, and genuine model-derived disease probability inference
for Diabetes, Heart Disease, and Chronic Kidney Disease (CKD) with Captum
Integrated Gradients explainability (XAI).
==============================================================================
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.train_multidisease_gnn import FullMultiDiseaseModel, FEATURES, TARGETS
from app.inference.ptbxl_inference import PTBXLInferenceService

logger = logging.getLogger("multidisease_inference")


class MultiDiseaseInferenceService:
    """
    Thread-safe, singleton multi-disease clinical inference service.
    """
    _instance: Optional["MultiDiseaseInferenceService"] = None

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        models_dir = PROJECT_ROOT / "models"
        
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (models_dir / "multidisease_gnn_best.pt")
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else (models_dir / "clinical_preprocessor.joblib")
        
        self.features = FEATURES
        self.targets = TARGETS
        
        self.preprocessor = None
        self.model = None
        self.ptbxl_service = PTBXLInferenceService.get_instance()
        
        self._load_preprocessor()
        self._load_model()

    @classmethod
    def get_instance(cls, **kwargs) -> "MultiDiseaseInferenceService":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _load_preprocessor(self):
        if self.preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("Loaded clinical preprocessor from %s", self.preprocessor_path)
            except Exception as e:
                logger.error("Error loading preprocessor: %s", e)
        else:
            logger.warning("Preprocessor path not found at %s", self.preprocessor_path)

    def _load_model(self):
        if not self.checkpoint_path.exists():
            logger.warning("Multi-Disease checkpoint not found at %s", self.checkpoint_path)
            self.model = None
            return
            
        self.model = FullMultiDiseaseModel(
            input_dim=len(self.features),
            embedding_dim=64,
            gnn_hidden=64,
            gnn_out=32,
            k_neighbors=5,
            dropout_rate=0.0
        ).to(self.device)
        
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
            
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Loaded MultiDiseaseGNN from %s", self.checkpoint_path)

    def preprocess_patient(self, data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts, formats, imputes, and standardizes patient clinical features.
        """
        # Encode sex: 1.0 = Male, 0.0 = Female
        gender = str(data.get("gender", "Male")).strip().lower()
        sex_num = 1.0 if gender in ["male", "m", "man", "1", "1.0"] else 0.0
        
        age = float(data.get("age", 50.0))
        height = float(data.get("height", 170.0))
        weight = float(data.get("weight", 70.0))
        
        # Calculate BMI if missing
        bmi_val = data.get("bmi")
        if bmi_val is not None and float(bmi_val) > 0:
            bmi = float(bmi_val)
        else:
            height_m = height / 100.0 if height > 0 else 1.70
            bmi = weight / (height_m * height_m) if height_m > 0 else 24.2
            
        systolic = float(data.get("systolic", data.get("systolic_bp", 120.0)))
        diastolic = float(data.get("diastolic", data.get("diastolic_bp", 80.0)))
        glucose = float(data.get("glucose", 95.0))
        hba1c = float(data.get("hba1c", 5.4))
        hdl = float(data.get("hdl", 50.0))
        total_chol = float(data.get("totalCholesterol", data.get("total_cholesterol", 190.0)))
        creatinine = float(data.get("creatinine", 0.9))
        bun = float(data.get("bun", 14.0))
        waist = float(data.get("waist", 85.0)) if data.get("waist") is not None else 85.0
        
        raw_dict = {
            "age": age,
            "sex": sex_num,
            "bmi": bmi,
            "waist": waist,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "hdl": hdl,
            "total_cholesterol": total_chol,
            "glucose": glucose,
            "hba1c": hba1c,
            "creatinine": creatinine,
            "bun": bun
        }
        
        raw_arr = np.array([[raw_dict[f] for f in self.features]], dtype=np.float32)
        
        if self.preprocessor is not None:
            imputed = self.preprocessor["imputer"].transform(raw_arr)
            scaled = self.preprocessor["scaler"].transform(imputed)
        else:
            scaled = raw_arr
            
        return scaled.astype(np.float32), raw_dict

    def _compute_integrated_gradients_xai(
        self,
        scaled_tensor: torch.Tensor,
        raw_dict: Dict[str, float],
        target_name: str
    ) -> List[Dict[str, Any]]:
        """
        Computes genuine Integrated Gradients attributions for a target disease logit.
        """
        self.model.eval()
        
        def forward_wrapper(x):
            logits_dict = self.model(x)
            return logits_dict[target_name].view(-1)
            
        ig = IntegratedGradients(forward_wrapper)
        c_in = scaled_tensor.to(self.device).clone().detach().requires_grad_(True)
        baseline = torch.zeros_like(c_in)
        
        attrs = ig.attribute(c_in, baselines=baseline, n_steps=40)
        attr_vals = attrs.detach().cpu().numpy().flatten()
        
        # Friendly feature names mapping
        name_map = {
            "age": "Age",
            "sex": "Sex",
            "bmi": "BMI",
            "waist": "Waist Circumference",
            "systolic_bp": "Systolic Blood Pressure",
            "diastolic_bp": "Diastolic Blood Pressure",
            "hdl": "HDL Cholesterol",
            "total_cholesterol": "Total Cholesterol",
            "glucose": "Fasting Glucose",
            "hba1c": "HbA1c",
            "creatinine": "Serum Creatinine",
            "bun": "Blood Urea Nitrogen (BUN)"
        }
        
        results = []
        for i, fname in enumerate(self.features):
            val = raw_dict.get(fname, 0.0)
            attr = float(attr_vals[i])
            direction = "pushes toward higher risk" if attr > 0 else "pushes toward lower risk"
            results.append({
                "feature": name_map.get(fname, fname),
                "feature_key": fname,
                "value": round(float(val), 2),
                "attribution": round(attr, 5),
                "importance": round(abs(attr), 5),
                "impact": f"{direction} ({attr:+.4f})",
                "direction": "toward_risk" if attr > 0 else "away_from_risk"
            })
            
        # Sort by absolute importance descending
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes real Multi-Disease GNN prediction + PTB-XL ECG analysis.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise FileNotFoundError(f"Multi-Disease checkpoint not found at {self.checkpoint_path}")
                
        scaled_vec, raw_dict = self.preprocess_patient(patient_data)
        scaled_tensor = torch.tensor(scaled_vec, dtype=torch.float32, device=self.device)
        
        # 1. Multi-Disease Model Forward Pass
        with torch.no_grad():
            logits_dict = self.model(scaled_tensor)
            probs = {
                t: float(torch.sigmoid(logits_dict[t]).item())
                for t in self.targets
            }
            
        # 2. XAI for each disease
        disease_results = {}
        for t in self.targets:
            prob = probs[t]
            if prob >= 0.50:
                risk_level = "High Risk" if prob >= 0.65 else "Moderate Risk"
                status = "Positive Risk"
            else:
                risk_level = "Low Risk" if prob < 0.35 else "Moderate Risk"
                status = "Normal / Low Risk"
                
            confidence = round((prob if prob >= 0.5 else (1.0 - prob)) * 100, 1)
            
            xai_attrs = self._compute_integrated_gradients_xai(scaled_tensor, raw_dict, target_name=t)
            
            disease_results[t] = {
                "disease": t.replace("_", " ").title(),
                "probability": round(prob, 4),
                "probability_pct": round(prob * 100, 1),
                "risk_level": risk_level,
                "status": status,
                "confidence": confidence,
                "top_drivers": xai_attrs[:4],
                "attributions": xai_attrs
            }
            
        # 3. PTB-XL ECG Assessment
        ptbxl_result = self.ptbxl_service.predict(patient_data, explain=True)
        
        # Overall Summary status
        highest_disease = max(self.targets, key=lambda k: probs[k])
        overall_prob = probs[highest_disease]
        overall_risk = disease_results[highest_disease]["risk_level"]
        
        return {
            "status": "success",
            "prediction": "Abnormal" if overall_prob >= 0.50 else "Normal",
            "probability": round(overall_prob, 4),
            "confidence": disease_results[highest_disease]["confidence"],
            "risk_level": overall_risk,
            "model": "Clinical Multi-Disease GNN (MultiDiseaseGNN) + Independent PTB-XL ECG",
            "disclaimer": "This AI-generated result is for clinical decision support research and is not a definitive medical diagnosis.",
            "diseases": disease_results,
            "ecg_assessment": {
                "prediction": ptbxl_result["prediction"],
                "probability": ptbxl_result["probability"],
                "confidence": ptbxl_result["confidence"],
                "risk_level": ptbxl_result["risk_level"],
                "clinical_explanation": ptbxl_result.get("clinical_explanation"),
                "ecg_explanation": ptbxl_result.get("ecg_explanation")
            },
            "graph_explanation": {
                "nodes": 1,
                "edges": 0,
                "neighbors": 0,
                "message_passing": False,
                "cross_patient_message_passing": False,
                "architecture": "MultiDiseaseGNN",
                "training_graph": "k-NN Patient Similarity Graph (k=5, Cosine Metric)",
                "inference_graph": "Isolated Single-Patient Node (N=1, E=0)",
                "explanation": "Single-patient inference uses an isolated graph node (Nodes=1, Edges=0). No cross-patient message passing occurs for this prediction."
            }
        }
