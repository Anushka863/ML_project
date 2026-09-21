"""
Unit and Integration Tests for FastAPI Backend Endpoints
=========================================================
Verifies:
1. /health endpoint returns 200 OK.
2. /model-info endpoint returns PTB-XL Multimodal GNN specs.
3. /predict endpoint accepts valid patient request and returns PredictionResponse.
4. /predict endpoint rejects invalid inputs (e.g. negative age, missing required fields).
5. Output contains prediction, probability, risk_level, model, and disclaimer.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_payload = {
            "age": 58,
            "gender": "Male",
            "height": 175.0,
            "weight": 82.0,
            "bmi": 26.8,
            "systolic": 138.0,
            "diastolic": 88.0,
            "glucose": 126.0,
            "hba1c": 6.8,
            "hdl": 42.0,
            "totalCholesterol": 215.0,
            "creatinine": 1.2,
            "bun": 18.0,
            "waist": 96.0
        }

    def test_01_health_endpoint(self):
        """GET /health returns healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_02_model_info_endpoint(self):
        """GET /model-info returns Phase 5 model architecture details."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("PTB-XL Multimodal", data["model_name"])
        self.assertIn("architecture", data)
        self.assertIn("test_performance", data)

    def test_03_predict_valid_patient(self):
        """POST /predict executes inference if checkpoint exists or returns 503 if checkpoint not available."""
        response = self.client.post("/predict", json=self.valid_payload)
        checkpoint_exists = (PROJECT_ROOT / "models" / "ptbxl_multimodal" / "best_model.pt").exists()
        if checkpoint_exists:
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertIn(data["prediction"], ["Normal", "Abnormal"])
            self.assertGreaterEqual(data["probability"], 0.0)
            self.assertLessEqual(data["probability"], 1.0)
            self.assertIn(data["risk_level"], ["Low Risk", "Moderate Risk", "High Risk"])
            self.assertEqual(data["model"], "PTB-XL Multimodal GNN")
            self.assertIn("disclaimer", data)
            self.assertIn("predictions", data)
        else:
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertIn("detail", data)
            self.assertIn("checkpoint not available", data["detail"].lower())

    def test_04_predict_invalid_missing_fields_rejected(self):
        """POST /predict returns 422 Unprocessable Entity for missing required fields."""
        invalid_payload = {"age": 58} # Missing gender, systolic, glucose, etc.
        response = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_05_predict_invalid_negative_age_rejected(self):
        """POST /predict returns 422 for invalid negative age."""
        bad_payload = dict(self.valid_payload)
        bad_payload["age"] = -5
        response = self.client.post("/predict", json=bad_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
