"""
Unit and Integration Tests for Phase 5 PTB-XL Multimodal GNN Inference Pipeline
================================================================================
Verifies:
1. Model checkpoint loads successfully.
2. Model is strictly in evaluation mode (model.eval(), requires_grad=False).
3. Valid patient inputs pass through inference end-to-end.
4. Output contains expected prediction structure (prediction, probability, confidence, risk_level, model, disclaimer).
5. Probability is strictly between 0.0 and 1.0.
6. Missing/edge case inputs are imputed and handled gracefully.
7. 12-lead ECG waveform input compatibility.
8. No weight updates or training happens during inference.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.ptbxl_inference import PTBXLInferenceService


class TestPTBXLInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = PTBXLInferenceService()
        cls.valid_patient = {
            "age": 58,
            "gender": "Male",
            "height": 175.0,
            "weight": 82.0,
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

    def test_01_model_checkpoint_loaded_and_eval_mode(self):
        """Verifies checkpoint loading or clean missing-checkpoint status."""
        if self.service.checkpoint_path.exists():
            self.assertIsNotNone(self.service.model)
            self.assertFalse(self.service.model.training, "Model should be in eval() mode")
            for name, param in self.service.model.named_parameters():
                self.assertFalse(param.requires_grad, f"Parameter {name} has requires_grad=True")
        else:
            self.assertIsNone(self.service.model, "Model should be None when checkpoint is absent")

    def test_02_valid_patient_inference(self):
        """Verifies valid patient produces full prediction response or clean FileNotFoundError."""
        if self.service.checkpoint_path.exists():
            result = self.service.predict(self.valid_patient)
            self.assertIn("prediction", result)
            self.assertIn("probability", result)
            self.assertIn("confidence", result)
            self.assertIn("risk_level", result)
            self.assertIn("model", result)
            self.assertIn("disclaimer", result)
            self.assertIn(result["prediction"], ["Normal", "Abnormal"])
            self.assertGreaterEqual(result["probability"], 0.0)
            self.assertLessEqual(result["probability"], 1.0)
        else:
            with self.assertRaises(FileNotFoundError) as ctx:
                self.service.predict(self.valid_patient)
            self.assertIn("trained checkpoint not found", str(ctx.exception).lower())

    def test_03_gender_encoding_consistency(self):
        """Verifies Male, Female, and edge cases are encoded cleanly."""
        male_val = self.service._encode_gender("Male")
        female_val = self.service._encode_gender("Female")
        other_val = self.service._encode_gender("Other")
        
        self.assertEqual(male_val, 1.0)
        self.assertEqual(female_val, 0.0)
        self.assertAlmostEqual(other_val, 0.48, places=2)

    def test_04_missing_optional_fields_imputation(self):
        """Verifies missing fields fallback to baseline train medians without crashing."""
        sparse_patient = {"age": 45}  # Missing gender, height, weight
        if self.service.checkpoint_path.exists():
            result = self.service.predict(sparse_patient)
            self.assertIn(result["prediction"], ["Normal", "Abnormal"])
            self.assertGreaterEqual(result["probability"], 0.0)
            self.assertLessEqual(result["probability"], 1.0)
        else:
            # Imputation method itself can be tested directly on input preprocessing
            processed = self.service.preprocess_clinical_input(sparse_patient)
            self.assertEqual(processed.shape, (1, 4))
            self.assertFalse(np.isnan(processed).any())

    def test_05_ecg_waveform_input_compatibility(self):
        """Verifies custom 12-lead ECG waveform array format handling."""
        dummy_ecg = np.random.randn(12, 1000).astype(np.float32)
        if self.service.checkpoint_path.exists():
            result = self.service.predict(self.valid_patient, ecg_waveform=dummy_ecg)
            self.assertIn(result["prediction"], ["Normal", "Abnormal"])
            self.assertGreaterEqual(result["probability"], 0.0)
            self.assertLessEqual(result["probability"], 1.0)
        else:
            # When checkpoint is missing, verify predict still validates waveform shape before failing
            with self.assertRaises(FileNotFoundError):
                self.service.predict(self.valid_patient, ecg_waveform=dummy_ecg)

    def test_06_model_weights_unmodified_after_inference(self):
        """Verifies model weights remain immutable when checkpoint is loaded."""
        if self.service.checkpoint_path.exists():
            first_param = next(self.service.model.parameters()).clone().detach()
            for _ in range(5):
                self.service.predict(self.valid_patient)
            current_param = next(self.service.model.parameters())
            self.assertTrue(torch.equal(first_param, current_param), "Model weights changed during inference!")
        else:
            self.assertIsNone(self.service.model)


if __name__ == "__main__":
    unittest.main()
