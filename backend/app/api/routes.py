"""
FastAPI Routes module.
Phase 15 — API endpoints for health check, model info, and patient risk prediction.
"""
from fastapi import APIRouter, HTTPException, Depends
from backend.app.api.schemas import PatientAssessmentRequest, PredictionResponse
from backend.app.services.prediction_service import ClinicalPredictionService

router = APIRouter()

# Global singleton service instance
_prediction_service: ClinicalPredictionService = None


def get_prediction_service() -> ClinicalPredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = ClinicalPredictionService()
    return _prediction_service


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Explainable Multi-Disease Clinical Decision Support System",
        "version": "1.0.0"
    }


@router.get("/model-info")
def model_info():
    """Model architecture and dataset info endpoint."""
    return {
        "model_name": "Multi-Disease Patient Similarity Graph Neural Network (GNN)",
        "diseases_covered": ["Diabetes Mellitus", "Heart Disease", "Chronic Kidney Disease (CKD)"],
        "architecture": {
            "tabular_encoder": "MLP (64-dim embedding)",
            "graph_type": "k-NN Cosine Similarity Graph (k=5)",
            "gnn_backbone": "MultiDiseaseGNN (GCN/GAT layers)",
            "xai_engine": "Tabular Feature Attributions + Grad-CAM"
        },
        "supported_features": [
            "age", "gender", "height", "weight", "bmi",
            "systolic", "diastolic", "glucose", "hba1c",
            "hdl", "totalCholesterol", "creatinine", "bun", "waist"
        ]
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_patient_risk(
    request: PatientAssessmentRequest,
    service: ClinicalPredictionService = Depends(get_prediction_service)
):
    """
    Patient risk prediction endpoint.
    Accepts clinical assessment parameters, executes preprocessing, graph embedding,
    GNN inference, and returns risk scores with XAI feature attributions.
    """
    try:
        response = service.predict_patient_risk(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction service error: {str(e)}")
