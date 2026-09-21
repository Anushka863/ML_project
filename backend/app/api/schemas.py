"""
API Pydantic Schemas for Request & Response Validation.
Phase 15 — Matches frontend assessment form fields exactly.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List


class PatientAssessmentRequest(BaseModel):
    # Basic Info
    age: float = Field(..., gt=0, lt=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    height: Optional[float] = Field(None, gt=0, description="Height in cm")
    weight: Optional[float] = Field(None, gt=0, description="Weight in kg")
    bmi: float = Field(..., gt=0, lt=100, description="Body Mass Index")

    # Vital Signs
    systolic: float = Field(..., gt=0, lt=300, description="Systolic Blood Pressure (mmHg)")
    diastolic: float = Field(..., gt=0, lt=200, description="Diastolic Blood Pressure (mmHg)")

    # Lab Info
    glucose: float = Field(..., ge=0, description="Fasting Blood Glucose (mg/dL)")
    hba1c: float = Field(..., ge=0, description="HbA1c percentage (%)")
    hdl: float = Field(..., ge=0, description="HDL Cholesterol (mg/dL)")
    totalCholesterol: float = Field(..., ge=0, description="Total Cholesterol (mg/dL)")
    creatinine: float = Field(..., ge=0, description="Serum Creatinine (mg/dL)")
    bun: float = Field(..., ge=0, description="Blood Urea Nitrogen (mg/dL)")

    # Additional
    waist: Optional[float] = Field(None, ge=0, description="Waist Circumference (cm)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 58,
                "gender": "Male",
                "height": 175,
                "weight": 82,
                "bmi": 26.8,
                "systolic": 138,
                "diastolic": 88,
                "glucose": 126,
                "hba1c": 6.8,
                "hdl": 42,
                "totalCholesterol": 215,
                "creatinine": 1.2,
                "bun": 18,
                "waist": 96
            }
        }
    )


class SingleDiseasePrediction(BaseModel):
    risk_score: float
    prediction: str
    risk_level: str


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    importance: float
    impact: str


class PredictionResponse(BaseModel):
    status: str = "success"
    prediction: str = Field(..., description="Normal or Abnormal prediction from PTB-XL Multimodal GNN")
    probability: float = Field(..., ge=0.0, le=1.0, description="Model sigmoid probability for abnormality")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage")
    risk_level: str = Field(..., description="Risk category: Low Risk, Moderate Risk, or High Risk")
    model: str = Field(default="PTB-XL Multimodal GNN", description="Model identifier")
    disclaimer: str = Field(
        default="This AI-generated result is for research/educational purposes and is not a medical diagnosis.",
        description="Medical disclaimer"
    )
    predictions: Optional[Dict[str, SingleDiseasePrediction]] = None
    clinical_explanation: Optional[List[FeatureAttribution]] = None
    image_explanation: Optional[Dict[str, Any]] = None
    graph_explanation: Optional[Dict[str, Any]] = None

