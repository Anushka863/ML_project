"""
Prediction Service module for FastAPI Backend.
Phase 15 — Connects Preprocessor -> Clinical Encoder -> Patient Graph -> GNN Model -> XAI.
"""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.preprocessing.clinical_preprocessor import ClinicalPreprocessor
from app.models.clinical_encoder import ClinicalEncoder
from app.graph.patient_graph import PatientGraphBuilder
from app.models.gnn_model import MultiDiseaseGNN
from app.explainability.tabular_xai import TabularXAI
from backend.app.api.schemas import PatientAssessmentRequest, PredictionResponse, SingleDiseasePrediction, FeatureAttribution


class ClinicalPredictionService:
    def __init__(self):
        # Initialize component pipelines
        self.preprocessor = ClinicalPreprocessor(random_state=42)
        
        # Fit baseline preprocessor schema for feature names
        dummy_df = pd.DataFrame([{
            "age": 50.0, "sex": "Male", "bmi": 25.0, "waist": 85.0,
            "systolic_bp": 120.0, "diastolic_bp": 80.0, "hdl": 50.0,
            "total_cholesterol": 190.0, "glucose": 95.0, "hba1c": 5.4,
            "creatinine": 0.9, "bun": 14.0, "egfr": 90.0
        }])
        self.preprocessor.fit(dummy_df)
        
        input_dim = len(self.preprocessor.fitted_feature_names)
        self.clinical_encoder = ClinicalEncoder(input_dim=input_dim, embedding_dim=64)
        self.graph_builder = PatientGraphBuilder(k_neighbors=5, metric="cosine")
        self.gnn_model = MultiDiseaseGNN(in_channels=64, hidden_channels=32, out_channels=16)
        self.xai_engine = TabularXAI(feature_names=self.preprocessor.fitted_feature_names)
        
        self.clinical_encoder.eval()
        self.gnn_model.eval()

    def _convert_request_to_dataframe(self, req: PatientAssessmentRequest) -> pd.DataFrame:
        """Map frontend API request dictionary to preprocessor input DataFrame."""
        egfr_val = max(15.0, 141.0 * min((req.creatinine / 0.9), 1.0)**(-0.411) * (0.993**req.age))
        
        data_dict = {
            "age": req.age,
            "sex": req.gender,
            "bmi": req.bmi,
            "waist": req.waist if req.waist is not None else 85.0,
            "systolic_bp": req.systolic,
            "diastolic_bp": req.diastolic,
            "hdl": req.hdl,
            "total_cholesterol": req.totalCholesterol,
            "glucose": req.glucose,
            "hba1c": req.hba1c,
            "creatinine": req.creatinine,
            "bun": req.bun,
            "egfr": round(egfr_val, 1)
        }
        return pd.DataFrame([data_dict])

    def predict_patient_risk(self, req: PatientAssessmentRequest) -> PredictionResponse:
        """
        Full inference pipeline:
        Request -> Preprocessing -> ClinicalEncoder -> PatientGraph -> GNN -> XAI -> Response
        """
        df_patient = self._convert_request_to_dataframe(req)
        
        # 1. Clinical Preprocessing
        x_processed = self.preprocessor.transform(df_patient)
        
        # 2. Clinical Encoder Forward Pass
        with torch.no_grad():
            x_tensor = torch.tensor(x_processed, dtype=torch.float32)
            embedding = self.clinical_encoder(x_tensor)
            
            # 3. Patient Graph Construction (Single node edge-less graph for single-patient request)
            edge_index, edge_attr, node_features = self.graph_builder.build_graph(embedding)
            
            # 4. GNN Multi-Disease Risk Inference
            logits = self.gnn_model(node_features, edge_index)
            
        predictions = {}
        for disease, logit_tensor in logits.items():
            prob = float(torch.sigmoid(logit_tensor).item())
            
            if prob >= 0.65:
                level = "High Risk"
            elif prob >= 0.40:
                level = "Moderate Risk"
            else:
                level = "Low Risk"
                
            predictions[disease] = SingleDiseasePrediction(
                risk_score=round(prob, 4),
                prediction=f"{disease.replace('_', ' ').title()}: {level}",
                risk_level=level
            )

        # 5. Tabular XAI Attributions
        raw_xai = self.xai_engine.compute_feature_attributions(
            model=self.clinical_encoder,
            input_vector=x_processed,
            target_disease="diabetes"
        )
        
        clinical_xai = [FeatureAttribution(**item) for item in raw_xai[:5]]

        return PredictionResponse(
            status="success",
            predictions=predictions,
            clinical_explanation=clinical_xai,
            image_explanation=None,
            graph_explanation={
                "message": "Single patient graph node evaluated against clinical GNN embedding baseline."
            }
        )
