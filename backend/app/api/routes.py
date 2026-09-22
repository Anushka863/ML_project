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
        "model_name": "Clinical Multi-Disease GNN & Independent PTB-XL Multimodal ECG",
        "multidisease_gnn": {
            "model_name": "Clinical Multi-Disease Graph Neural Network (MultiDiseaseGNN)",
            "diseases_covered": ["Diabetes", "Heart Disease", "Chronic Kidney Disease (CKD)"],
            "checkpoint": "models/multidisease_gnn_best.pt",
            "input_features": [
                "age", "sex", "bmi", "waist", "systolic_bp", "diastolic_bp",
                "hdl", "total_cholesterol", "glucose", "hba1c", "creatinine", "bun"
            ],
            "feature_count": 12,
            "architecture": {
                "clinical_encoder": "ClinicalEncoder MLP (12 -> 64-dim)",
                "graph_builder": "k-NN Patient Similarity Graph (k=5, Cosine Metric during training)",
                "gnn_backbone": "MultiDiseaseGNN (SimpleGNNConv 64 -> 64 -> 32-dim)",
                "prediction_heads": "Multi-Head Binary Logits (32 -> 16 -> 1 per disease)"
            },
            "test_performance": {
                "diabetes_roc_auc": 0.9470,
                "heart_disease_roc_auc": 0.7388,
                "ckd_roc_auc": 0.9896
            },
            "inference_graph_state": {
                "nodes": 1,
                "edges": 0,
                "cross_patient_message_passing": False,
                "note": "Single-patient web inference uses an isolated graph node (N=1, E=0). No cross-patient message passing occurs for single-patient predictions."
            }
        },
        "ptbxl_ecg_branch": {
            "model_name": "Independent 12-Lead ECG Analysis (PTB-XL Multimodal GNN)",
            "branch_type": "Separate Independent Diagnostic Branch",
            "diseases_covered": ["ECG Arrhythmia & Diagnostic Abnormality"],
            "checkpoint": "models/ptbxl_multimodal/best_model.pt",
            "architecture": {
                "clinical_encoder": "Clinical MLP (4 -> 64-dim embedding: age, sex, height, weight)",
                "ecg_encoder": "1D CNN Waveform Encoder (12 leads -> 128-dim embedding)",
                "multimodal_fusion": "Multimodal Fusion Layer (128-dim)",
                "graph_type": "k-NN Patient Similarity Graph (k=5, Cosine Metric)",
                "gnn_backbone": "MultiDiseaseGNN (SimpleGNNConv Message Passing)"
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
        },
        "architecture": {
            "clinical_encoder": "ClinicalEncoder MLP (12 -> 64-dim)",
            "multidisease_model": "Clinical Multi-Disease GNN (models/multidisease_gnn_best.pt)",
            "ecg_model": "Independent 12-Lead ECG GNN (models/ptbxl_multimodal/best_model.pt)",
            "multimodal_fusion": "Multimodal Fusion Layer (128-dim for ECG branch)"
        },
        "test_performance": {
            "diabetes_roc_auc": "0.9470",
            "heart_disease_roc_auc": "0.7388",
            "ckd_roc_auc": "0.9896",
            "ecg_abnormality_roc_auc": "0.8252"
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
