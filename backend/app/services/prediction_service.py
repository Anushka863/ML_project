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

from app.inference.multidisease_inference import MultiDiseaseInferenceService
from backend.app.api.schemas import (
    PatientAssessmentRequest,
    PredictionResponse,
    DiseaseDetail,
    SingleDiseasePrediction,
    FeatureAttribution,
    AdditionalClinicalMarker
)

logger = logging.getLogger("prediction_service")


class ClinicalPredictionService:
    def __init__(self):
        # Initialize Multi-Disease GNN Inference Service (cached singleton)
        self.multidisease_service = MultiDiseaseInferenceService.get_instance()
        logger.info("ClinicalPredictionService initialized with MultiDiseaseInferenceService.")

    def predict_patient_risk(self, req: PatientAssessmentRequest) -> PredictionResponse:
        """
        Full inference pipeline:
        Request -> Unified Preprocessing -> MultiDiseaseGNN Inference (Diabetes, Heart Disease, CKD) + PTB-XL ECG -> Response
        """
        patient_dict = req.model_dump()
        result = self.multidisease_service.predict(patient_dict)
        
        # Build disease details mapping
        disease_details = {}
        for d_key, d_val in result["diseases"].items():
            disease_details[d_key] = DiseaseDetail(
                disease=d_val["disease"],
                probability=d_val["probability"],
                probability_pct=d_val["probability_pct"],
                risk_level=d_val["risk_level"],
                status=d_val["status"],
                confidence=d_val["confidence"],
                top_drivers=d_val["top_drivers"],
                attributions=d_val["attributions"]
            )
            
        # Demographic clinical feature attributions from PTB-XL ECG branch (Age, Sex, Height, Weight)
        ecg_data = result.get("ecg_assessment", {})
        clinical_xai = []
        if ecg_data and "clinical_explanation" in ecg_data and ecg_data["clinical_explanation"]:
            for item in ecg_data["clinical_explanation"]:
                val = float(item.get("input_value", item.get("value", 0.0)))
                attr = float(item.get("attribution", item.get("importance", 0.0)))
                clinical_xai.append(
                    FeatureAttribution(
                        feature=item["feature"],
                        value=val,
                        importance=round(abs(attr), 5),
                        impact=item.get("impact", f"attribution ({attr:+.4f})"),
                        direction=item.get("direction"),
                        attribution=round(attr, 5)
                    )
                )
        else:
            highest_disease_key = max(result["diseases"].keys(), key=lambda k: result["diseases"][k]["probability"])
            for item in result["diseases"][highest_disease_key]["attributions"]:
                val = float(item.get("value", item.get("input_value", 0.0)))
                attr = float(item.get("attribution", item.get("importance", 0.0)))
                clinical_xai.append(
                    FeatureAttribution(
                        feature=item["feature"],
                        value=val,
                        importance=round(abs(attr), 5),
                        impact=item.get("impact", f"attribution ({attr:+.4f})"),
                        direction=item.get("direction"),
                        attribution=round(attr, 5)
                    )
                )

        # Additional Clinical Markers for quick reference (distinct from model predictions)
        ref_note = "Clinical reference marker; displayed for contextual reference and NOT an input feature to the PTB-XL ECG GNN model; does NOT calculate or override the machine learning model prediction."
        additional_info = [
            AdditionalClinicalMarker(
                marker="Blood Pressure",
                value=float(req.systolic),
                unit="mmHg",
                reference_range="< 120/80 mmHg",
                clinical_status="Elevated" if req.systolic >= 130 or req.diastolic >= 85 else "Optimal",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="BMI",
                value=float(req.bmi),
                unit="kg/m²",
                reference_range="18.5 - 24.9 kg/m²",
                clinical_status="Elevated" if req.bmi >= 25.0 else "Normal",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="Fasting Glucose",
                value=float(req.glucose),
                unit="mg/dL",
                reference_range="70 - 99 mg/dL",
                clinical_status="Elevated" if req.glucose >= 100 else "Normal",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="HbA1c",
                value=float(req.hba1c),
                unit="%",
                reference_range="< 5.7%",
                clinical_status="Elevated" if req.hba1c >= 5.7 else "Normal",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="Total Cholesterol",
                value=float(req.totalCholesterol),
                unit="mg/dL",
                reference_range="< 200 mg/dL",
                clinical_status="Borderline / High" if req.totalCholesterol >= 200 else "Desirable",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="Serum Creatinine",
                value=float(req.creatinine),
                unit="mg/dL",
                reference_range="0.7 - 1.3 mg/dL",
                clinical_status="High" if req.creatinine > 1.3 else "Normal",
                note=ref_note
            ),
            AdditionalClinicalMarker(
                marker="Blood Urea Nitrogen (BUN)",
                value=float(req.bun),
                unit="mg/dL",
                reference_range="7 - 20 mg/dL",
                clinical_status="High" if req.bun > 20 else "Normal",
                note=ref_note
            )
        ]

        # Backward-compatible predictions dictionary
        predictions = {
            t: SingleDiseasePrediction(
                risk_score=result["diseases"][t]["probability"],
                prediction=f"{result['diseases'][t]['disease']} Risk: {result['diseases'][t]['risk_level']}",
                risk_level=result["diseases"][t]["risk_level"]
            )
            for t in result["diseases"]
        }

        ecg_data = result.get("ecg_assessment", {})
        
        return PredictionResponse(
            status="success",
            prediction=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            model=result["model"],
            disclaimer=result["disclaimer"],
            diseases=disease_details,
            ecg_assessment=ecg_data,
            predictions=predictions,
            clinical_explanation=clinical_xai,
            image_explanation=None,
            graph_explanation=result.get("graph_explanation"),
            ecg_explanation=ecg_data.get("ecg_explanation"),
            xai={"diseases": result["diseases"], "ecg": ecg_data.get("ecg_explanation")},
            additional_clinical_info=additional_info
        )
