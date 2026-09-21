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
        "model_name": "PTB-XL Multimodal Graph Neural Network (GNN)",
        "diseases_covered": ["ECG Arrhythmia & Diagnostic Abnormality", "Cardiovascular Disease"],
        "architecture": {
            "clinical_encoder": "Clinical MLP (4 -> 64-dim embedding)",
            "ecg_encoder": "1D CNN Waveform Encoder (12 leads -> 128-dim embedding)",
            "multimodal_fusion": "Multimodal Fusion Layer (128-dim)",
            "graph_type": "k-NN Patient Similarity Graph (k=5, Cosine Metric)",
            "gnn_backbone": "MultiDiseaseGNN (GCN/GAT Message Passing)",
            "checkpoint": "models/ptbxl_multimodal/best_model.pt"
        },
        "supported_features": [
            "age", "gender", "height", "weight", "bmi",
            "systolic", "diastolic", "glucose", "hba1c",
            "hdl", "totalCholesterol", "creatinine", "bun", "waist"
        ],
        "test_performance": {
            "accuracy": "72.71%",
            "precision": "0.8628",
            "recall": "0.6296",
            "specificity": "0.8617",
            "f1_score": "0.7280",
            "roc_auc": "0.8252"
        }
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
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Trained checkpoint not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction service error: {str(e)}")
