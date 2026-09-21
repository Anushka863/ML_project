"""
Prediction Service module for FastAPI Backend.
Connects Preprocessor -> Multimodal Representation -> Patient Graph -> Phase 5 PTB-XL GNN Model -> Response.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.inference.ptbxl_inference import PTBXLInferenceService
from backend.app.api.schemas import (
    PatientAssessmentRequest,
    PredictionResponse,
    SingleDiseasePrediction,
    FeatureAttribution
)

logger = logging.getLogger("prediction_service")


class ClinicalPredictionService:
    def __init__(self):
        # Initialize Phase 5 PTB-XL Multimodal GNN Inference Service (cached singleton)
        self.ptbxl_service = PTBXLInferenceService.get_instance()
        logger.info("ClinicalPredictionService initialized with PTBXLInferenceService.")

    def predict_patient_risk(self, req: PatientAssessmentRequest) -> PredictionResponse:
        """
        Full inference pipeline:
        Request -> PTB-XL Preprocessing -> Multimodal Representation -> Patient Graph -> GNN -> Response
        """
        # 1. Convert request pydantic model to dictionary
        patient_dict = req.model_dump()
        
        # 2. Run leak-free PTB-XL Multimodal GNN inference with genuine model-derived XAI
        ptbxl_result = self.ptbxl_service.predict(patient_dict, explain=True)
        xai_data = ptbxl_result.get("xai", {})

        # 3. Derive genuine model clinical feature attributions (Age, Sex, Height, Weight)
        clinical_xai = []
        raw_clinical_xai = xai_data.get("clinical_features", [])
        for item in raw_clinical_xai:
            fname = item["feature"].capitalize()
            attr_val = float(item["attribution"])
            val = float(item["input_value"])
            direction = item["direction"]
            clinical_xai.append(
                FeatureAttribution(
                    feature=fname,
                    value=val,
                    importance=round(abs(attr_val), 5),
                    impact=f"pushes {'toward abnormal' if attr_val > 0 else 'away from abnormal'} ({attr_val:+.4f})",
                    direction=direction,
                    attribution=round(attr_val, 5)
                )
            )

        # 4. Separate non-model clinical markers with explicit transparency note
        additional_info = [
            {
                "marker": "Blood Pressure",
                "value": float(req.systolic),
                "unit": "mmHg",
                "reference_range": "< 120/80 mmHg",
                "clinical_status": "Elevated" if req.systolic >= 130 or req.diastolic >= 85 else "Optimal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "BMI",
                "value": float(req.bmi),
                "unit": "kg/m²",
                "reference_range": "18.5 - 24.9 kg/m²",
                "clinical_status": "Elevated" if req.bmi >= 25.0 else "Normal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "Fasting Glucose",
                "value": float(req.glucose),
                "unit": "mg/dL",
                "reference_range": "70 - 99 mg/dL",
                "clinical_status": "Elevated" if req.glucose >= 100 else "Normal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "HbA1c",
                "value": float(req.hba1c),
                "unit": "%",
                "reference_range": "< 5.7%",
                "clinical_status": "Elevated" if req.hba1c >= 5.7 else "Normal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "Total Cholesterol",
                "value": float(req.totalCholesterol),
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "clinical_status": "Borderline / High" if req.totalCholesterol >= 200 else "Desirable",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "Serum Creatinine",
                "value": float(req.creatinine),
                "unit": "mg/dL",
                "reference_range": "0.7 - 1.3 mg/dL",
                "clinical_status": "High" if req.creatinine > 1.3 else "Normal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            },
            {
                "marker": "Blood Urea Nitrogen (BUN)",
                "value": float(req.bun),
                "unit": "mg/dL",
                "reference_range": "7 - 20 mg/dL",
                "clinical_status": "High" if req.bun > 20 else "Normal",
                "note": "General clinical marker; NOT an input feature to the PTB-XL ECG GNN model."
            }
        ]

        # 5. Construct backward-compatible predictions dictionary
        predictions = {
            "ptbxl_multimodal_gnn": SingleDiseasePrediction(
                risk_score=ptbxl_result["probability"],
                prediction=f"ECG Abnormality Risk: {ptbxl_result['risk_level']}",
                risk_level=ptbxl_result["risk_level"]
            ),
            "heart_disease": SingleDiseasePrediction(
                risk_score=ptbxl_result["probability"],
                prediction=f"Cardiovascular Risk: {ptbxl_result['risk_level']}",
                risk_level=ptbxl_result["risk_level"]
            )
        }

        return PredictionResponse(
            status="success",
            prediction=ptbxl_result["prediction"],
            probability=ptbxl_result["probability"],
            confidence=ptbxl_result["confidence"],
            risk_level=ptbxl_result["risk_level"],
            model=ptbxl_result["model"],
            disclaimer=ptbxl_result["disclaimer"],
            predictions=predictions,
            clinical_explanation=clinical_xai,
            image_explanation=None,
            graph_explanation=ptbxl_result.get("graph_explanation"),
            ecg_explanation=ptbxl_result.get("ecg_explanation"),
            xai=xai_data,
            additional_clinical_info=additional_info
        )
