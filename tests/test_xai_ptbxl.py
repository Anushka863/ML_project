"""
==============================================================================
Phase 2 — Genuine Model-Derived XAI Verification Tests
==============================================================================
Validates:
1. Clinical attribution generation for consumed features (age, sex, height, weight).
2. ECG attribution generation with shape dynamically derived from input tensor.
3. Dimension conformity and zero NaN / zero Inf guarantees.
4. Integrated Gradients Completeness Property: sum(attributions) ≈ F(input) - F(baseline).
5. Model weight invariance (explainer leaves checkpoint parameters unmodified).
6. Strict checkpoint loading validation (model.load_state_dict(..., strict=True)).
7. Graph explanation transparency (isolated N=1 node, 0 edges, no message passing).
8. End-to-end real PTB-XL test sample evaluation with WFDB waveform.
9. Controlled input perturbation test documenting model/attribution response.
10. Backend API schema integration containing structured XAI payload.
11. SHA256 immutability check of the trained checkpoint.
==============================================================================
"""

import os
import sys
import hashlib
import unittest
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.train_ptbxl_multimodal import PTBXLMultimodalGNN
from app.inference.ptbxl_inference import PTBXLInferenceService
from explainability.ptbxl_explainer import PTBXLExplainer, PTBXL_CLINICAL_COLS
from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from backend.app.main import app

EXPECTED_CHECKPOINT_SHA256 = "48922961129a6c1a6d3594c0addfc29edc7fe90ee20f8d2a2c53e7624d0eac98"


class TestPhase2ModelDerivedXAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = PROJECT_ROOT / "models" / "ptbxl_multimodal" / "best_model.pt"
        cls.preprocessor_path = PROJECT_ROOT / "models" / "ptbxl_multimodal" / "preprocessor.joblib"
        cls.data_dir = PROJECT_ROOT / "data" / "ptbxl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

        # Verify checkpoint exists
        if not cls.checkpoint_path.exists():
            raise FileNotFoundError(f"Trained checkpoint not found at {cls.checkpoint_path}")

        # Instantiate inference service
        cls.service = PTBXLInferenceService.get_instance()
        cls.model = cls.service.model
        cls.client = TestClient(app)

    def test_01_checkpoint_immutability_sha256(self):
        """Verify checkpoint has NOT been retrained, replaced, or modified."""
        with open(self.checkpoint_path, "rb") as f:
            current_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            current_sha256,
            EXPECTED_CHECKPOINT_SHA256,
            f"Checkpoint SHA256 mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {current_sha256}"
        )

    def test_02_strict_checkpoint_loading(self):
        """Verify model checkpoint loads strictly without missing or unexpected keys."""
        test_model = PTBXLMultimodalGNN(
            clinical_in=4,
            ecg_channels=12,
            clinical_emb_dim=64,
            ecg_emb_dim=128,
            fused_dim=128,
            gnn_hidden=64,
            gnn_out=32,
            k_neighbors=5,
            dropout_rate=0.0
        )
        cp = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = cp["model_state_dict"] if isinstance(cp, dict) and "model_state_dict" in cp else cp
        load_result = test_model.load_state_dict(state_dict, strict=True)
        self.assertEqual(len(load_result.missing_keys), 0)
        self.assertEqual(len(load_result.unexpected_keys), 0)

    def test_03_clinical_attribution_generation(self):
        """Verify Integrated Gradients produces valid attributions for clinical features."""
        explainer = PTBXLExplainer(model=self.model, n_steps=25)
        c_tensor = torch.tensor([[0.5, 1.0, -0.2, 0.3]], dtype=torch.float32)
        e_tensor = torch.zeros((1, 12, 1000), dtype=torch.float32)

        res = explainer.explain(c_tensor, e_tensor)
        c_features = res["clinical_features"]

        self.assertEqual(len(c_features), 4)
        feature_names = [f["feature"] for f in c_features]
        self.assertEqual(feature_names, PTBXL_CLINICAL_COLS)

        for f in c_features:
            self.assertIn("attribution", f)
            self.assertIn("direction", f)
            self.assertIn(f["direction"], ["toward_abnormal", "toward_normal"])
            self.assertFalse(np.isnan(f["attribution"]))
            self.assertFalse(np.isinf(f["attribution"]))

    def test_04_ecg_attribution_shape_matches_input(self):
        """Verify ECG attribution shape is derived dynamically and matches input shape (excluding batch)."""
        explainer = PTBXLExplainer(model=self.model, n_steps=20)
        c_tensor = torch.zeros((1, 4), dtype=torch.float32)
        
        # Test standard (1, 12, 1000)
        e_tensor = torch.randn((1, 12, 1000), dtype=torch.float32)
        res = explainer.explain(c_tensor, e_tensor)
        
        expected_shape = list(e_tensor.shape[1:])
        actual_shape = res["ecg"]["attribution_shape"]
        self.assertEqual(actual_shape, expected_shape)
        self.assertEqual(len(res["ecg"]["lead_attributions"]), 12)
        self.assertEqual(len(res["ecg"]["temporal_attributions"]), 10)

    def test_05_no_nan_no_inf_guarantee(self):
        """Ensure all returned attributions and metrics are strictly finite."""
        c_in = torch.randn(1, 4)
        e_in = torch.randn(1, 12, 1000)
        explainer = PTBXLExplainer(model=self.model, n_steps=25)
        res = explainer.explain(c_in, e_in)

        # Check clinical
        for f in res["clinical_features"]:
            self.assertTrue(np.isfinite(f["attribution"]))
            self.assertTrue(np.isfinite(f["input_value"]))

        # Check ECG
        for lead, val in res["ecg"]["lead_attributions"].items():
            self.assertTrue(np.isfinite(val))
        for seg in res["ecg"]["temporal_attributions"]:
            self.assertTrue(np.isfinite(seg["mean_attribution"]))

        # Check completeness
        cc = res["completeness_check"]
        self.assertTrue(np.isfinite(cc["f_input_logit"]))
        self.assertTrue(np.isfinite(cc["sum_of_attributions"]))

    def test_06_integrated_gradients_completeness_property(self):
        """
        Verify the mathematical completeness axiom:
        sum(attributions) ≈ F(input) - F(baseline)
        within documented numerical tolerance (Gauss-Legendre approximation).
        """
        torch.manual_seed(101)
        c_in = torch.randn(1, 4) * 0.5
        e_in = torch.randn(1, 12, 1000) * 0.5

        explainer = PTBXLExplainer(model=self.model, n_steps=50)
        res = explainer.explain(c_in, e_in)
        check = res["completeness_check"]

        f_diff = check["f_input_minus_f_baseline"]
        sum_attr = check["sum_of_attributions"]
        abs_err = abs(sum_attr - f_diff)

        # Completeness should hold within reasonable tolerance (abs < 0.15 or rel < 10%)
        rel_err = (abs_err / (abs(f_diff) + 1e-6)) * 100
        self.assertTrue(
            abs_err < 0.15 or rel_err < 10.0,
            f"IG Completeness violated: sum_attr={sum_attr}, F(in)-F(base)={f_diff}, abs_err={abs_err}, rel_err={rel_err}%"
        )

    def test_07_model_weight_invariance(self):
        """Verify that running explainability passes leaves model weights completely unmodified."""
        initial_params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        c_in = torch.randn(1, 4)
        e_in = torch.randn(1, 12, 1000)
        explainer = PTBXLExplainer(model=self.model, n_steps=20)
        _ = explainer.explain(c_in, e_in)

        for k, v in self.model.named_parameters():
            self.assertTrue(
                torch.equal(initial_params[k], v),
                f"Model parameter '{k}' was mutated during XAI explanation!"
            )

    def test_08_graph_explanation_honest_n1(self):
        """Verify graph explanation reports true single-patient graph (N=1, 0 edges, no message passing)."""
        patient_data = {"age": 60, "gender": "Male", "height": 170, "weight": 75}
        result = self.service.predict(patient_data, explain=True)
        
        graph_xai = result["xai"]["graph"]
        self.assertEqual(graph_xai["nodes"], 1)
        self.assertEqual(graph_xai["edges"], 0)
        self.assertEqual(graph_xai["neighbors"], [])
        self.assertFalse(graph_xai["message_passing"])
        self.assertIn("No cross-patient message passing", graph_xai["explanation"])

    def test_09_real_ptbxl_test_patient_evaluation(self):
        """
        Evaluate full leak-free pipeline on a GENUINE PTB-XL test sample:
        real patient clinical row + real WFDB waveform -> trained checkpoint -> prediction + XAI.
        """
        meta_path = PROJECT_ROOT / "data" / "ptbxl" / "test_metadata.csv"
        self.assertTrue(meta_path.exists(), "test_metadata.csv must exist")
        df_test = pd.read_csv(meta_path)
        
        # Select first test record
        row = df_test.iloc[0]
        fn_lr = row["filename_lr"]
        target = int(row["target"])

        # Load genuine waveform
        preprocessor = PTBXLPreprocessor(str(self.data_dir))
        ecg_waveform = preprocessor.extract_waveform(fn_lr) # (12, 1000)
        self.assertEqual(ecg_waveform.shape, (12, 1000))
        self.assertGreater(np.count_nonzero(ecg_waveform), 0)

        # Build patient intake dict from real record
        patient_data = {
            "age": float(row["age"]),
            "sex": float(row["sex"]),
            "height": float(row["height"]),
            "weight": float(row["weight"])
        }

        # Run inference with XAI
        result = self.service.predict(patient_data, ecg_waveform=ecg_waveform, explain=True)
        self.assertIn("prediction", result)
        self.assertIn("probability", result)
        self.assertIn("xai", result)

        xai = result["xai"]
        self.assertEqual(len(xai["clinical_features"]), 4)
        self.assertEqual(xai["ecg"]["attribution_shape"], [12, 1000])
        self.assertTrue(xai["completeness_check"]["completeness_satisfied"])

    def test_10_controlled_perturbation_sanity_check(self):
        """
        Sanity Check:
        Run inference on a real sample, apply controlled input perturbation, and document
        the measurable changes in output logits and/or attributions.
        """
        meta_path = PROJECT_ROOT / "data" / "ptbxl" / "test_metadata.csv"
        df_test = pd.read_csv(meta_path)
        row = df_test.iloc[0]
        
        preprocessor = PTBXLPreprocessor(str(self.data_dir))
        ecg_orig = preprocessor.extract_waveform(row["filename_lr"])
        patient_orig = {"age": float(row["age"]), "sex": float(row["sex"]), "height": float(row["height"]), "weight": float(row["weight"])}

        res_orig = self.service.predict(patient_orig, ecg_waveform=ecg_orig, explain=True)

        # Controlled perturbation: add high-amplitude synthetic ST-elevation/depression shift to Lead II
        ecg_perturbed = ecg_orig.copy()
        ecg_perturbed[1, 200:400] += 3.0  # Lead II localized alteration

        res_perturbed = self.service.predict(patient_orig, ecg_waveform=ecg_perturbed, explain=True)

        prob_delta = abs(res_perturbed["probability"] - res_orig["probability"])
        lead2_attr_orig = res_orig["xai"]["ecg"]["lead_attributions"]["II"]
        lead2_attr_perturbed = res_perturbed["xai"]["ecg"]["lead_attributions"]["II"]
        lead2_attr_delta = abs(lead2_attr_perturbed - lead2_attr_orig)

        # Verify that altering input signal produces a non-identical computational response
        has_response = (prob_delta > 1e-4) or (lead2_attr_delta > 1e-4)
        self.assertTrue(
            has_response,
            f"Perturbation failed to elicit model response: prob_delta={prob_delta}, lead2_attr_delta={lead2_attr_delta}"
        )

    def test_11_api_response_schema_contains_xai_and_separated_markers(self):
        """Verify FastAPI /predict response contains model XAI and cleanly separated non-model clinical markers."""
        payload = {
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
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("xai", data)
        self.assertIn("clinical_explanation", data)
        self.assertIn("additional_clinical_info", data)

        # Verify clinical attributions only contain model inputs
        explained_features = [item["feature"].lower() for item in data["clinical_explanation"]]
        self.assertEqual(set(explained_features), {"age", "sex", "height", "weight"})

        # Verify non-model markers are listed under additional_clinical_info with disclaimer note
        non_model_markers = [item["marker"] for item in data["additional_clinical_info"]]
        self.assertIn("Blood Pressure", non_model_markers)
        self.assertIn("BMI", non_model_markers)
        self.assertIn("Total Cholesterol", non_model_markers)
        for item in data["additional_clinical_info"]:
            self.assertIn("NOT an input feature to the PTB-XL ECG GNN model", item["note"])


if __name__ == "__main__":
    unittest.main()
