"""
FastAPI Routes module.
Phase 15 — API endpoints for health check, model info, and patient risk prediction.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from backend.app.api.schemas import (
    PatientAssessmentRequest,
    PredictionResponse,
    ODIRAssessmentRequest,
    ODIRPredictionResponse
)
from backend.app.services.prediction_service import ClinicalPredictionService

logger = logging.getLogger("backend.api.routes")
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
        "odir_ophthalmic_branch": {
            "model_name": "ODIR-5K Ophthalmic Multimodal GNN (ODIRMultimodalGNN)",
            "branch_type": "Separate Independent Diagnostic Branch",
            "diseases_covered": [
                "Normal (N)",
                "Diabetes (D)",
                "Glaucoma (G)",
                "Cataract (C)",
                "AMD (A)",
                "Hypertension (H)",
                "Myopia (M)",
                "Other (O)"
            ],
            "checkpoint": "models/odir_multimodal/best_model.pt",
            "architecture": {
                "visual_encoder": "ResNet-18 Dual-Stream Bilateral Visual Encoder (64 -> 128-dim)",
                "demographic_encoder": "ClinicalEncoder MLP (2 -> 32-dim: Age, Sex)",
                "multimodal_fusion": "Multimodal Fusion Layer (128-dim)",
                "graph_type": "k-NN Patient Similarity Graph (k=5, Cosine Metric)",
                "gnn_backbone": "MultiDiseaseGNN (SimpleGNNConv Message Passing)",
                "prediction_heads": "8 Multi-Head Binary Logits (N, D, G, C, A, H, M, O)"
            },
            "input_features": ["age", "sex", "left_fundus_image", "right_fundus_image"]
        },
        "architecture": {
            "clinical_encoder": "ClinicalEncoder MLP (12 -> 64-dim)",
            "multidisease_model": "Clinical Multi-Disease GNN (models/multidisease_gnn_best.pt)",
            "ecg_model": "Independent 12-Lead ECG GNN (models/ptbxl_multimodal/best_model.pt)",
            "odir_model": "ODIR-5K Multimodal GNN (models/odir_multimodal/best_model.pt)",
            "multimodal_fusion": "Multimodal Fusion Layer (128-dim for ECG & Ophthalmic branches)"
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


@router.post("/predict-odir", response_model=ODIRPredictionResponse)
def predict_odir_risk(
    request: ODIRAssessmentRequest,
    service: ClinicalPredictionService = Depends(get_prediction_service)
):
    """
    ODIR-5K Ophthalmic Multimodal Risk Prediction endpoint.
    Accepts patient Age, Sex, and bilateral fundus images (Base64 strings or local file paths).
    Returns multi-label probabilities for N, D, G, C, A, H, M, O + bilateral Grad-CAM XAI.
    """
    import base64
    import io
    from PIL import Image

    # Validate Left image
    if request.left_image_b64:
        try:
            left_bytes = base64.b64decode(request.left_image_b64.split(",")[-1])
            left_img = Image.open(io.BytesIO(left_bytes))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid Left eye image data: {str(e)}")
    elif request.left_image_path:
        left_img = request.left_image_path
    else:
        raise HTTPException(status_code=422, detail="Missing Left eye fundus image. Both Left and Right eye images are required.")

    # Validate Right image
    if request.right_image_b64:
        try:
            right_bytes = base64.b64decode(request.right_image_b64.split(",")[-1])
            right_img = Image.open(io.BytesIO(right_bytes))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid Right eye image data: {str(e)}")
    elif request.right_image_path:
        right_img = request.right_image_path
    else:
        raise HTTPException(status_code=422, detail="Missing Right eye fundus image. Both Left and Right eye images are required.")

    try:
        response = service.predict_odir_risk(
            age=request.age,
            sex=request.sex,
            left_img=left_img,
            right_img=right_img
        )
        return response
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"ODIR trained checkpoint not available: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ODIR Prediction error: {str(e)}")


@router.post("/predict-odir/upload", response_model=ODIRPredictionResponse)
async def predict_odir_upload(
    age: float = Form(..., description="Patient Age"),
    sex: str = Form(..., description="Patient Sex (Male/Female)"),
    left_image: UploadFile = File(..., description="Left Eye Fundus Image"),
    right_image: UploadFile = File(..., description="Right Eye Fundus Image"),
    service: ClinicalPredictionService = Depends(get_prediction_service)
):
    """
    Multipart file upload endpoint for ODIR-5K Ophthalmic Assessment.
    """
    import io
    from PIL import Image

    if not left_image or not left_image.filename:
        raise HTTPException(status_code=422, detail="Missing Left eye fundus image file.")
    if not right_image or not right_image.filename:
        raise HTTPException(status_code=422, detail="Missing Right eye fundus image file.")

    # Read and open left image
    try:
        left_bytes = await left_image.read()
        left_img = Image.open(io.BytesIO(left_bytes))
        left_img.verify()
        left_img = Image.open(io.BytesIO(left_bytes))  # re-open after verify
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid or corrupted Left eye image file: {str(e)}")

    try:
        right_bytes = await right_image.read()
        right_img = Image.open(io.BytesIO(right_bytes))
        right_img.verify()
        right_img = Image.open(io.BytesIO(right_bytes))  # re-open after verify
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid or corrupted Right eye image file: {str(e)}")

    logger.info(
        f"ODIR assessment upload received: left_image='{left_image.filename}' ({len(left_bytes)} bytes), "
        f"right_image='{right_image.filename}' ({len(right_bytes)} bytes), age={age}, sex='{sex}'"
    )

    try:
        response = service.predict_odir_risk(
            age=age,
            sex=sex,
            left_img=left_img,
            right_img=right_img
        )
        return response
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"ODIR trained checkpoint not available: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ODIR Prediction error: {str(e)}")

