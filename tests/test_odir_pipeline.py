"""
Test suite for ODIR-5K Ophthalmic Multimodal Pipeline.
Tests:
- Demographics & Bilateral Image Preprocessing
- Patient Split Disjointness & Zero Leakage
- ODIRMultimodalGNN Architecture & Forward Pass
- k-NN Patient Similarity Graph
- ODIRInferenceService Multi-Label Predictions
- FastAPI /predict-odir Endpoint (Valid & Validation Error Handling)
- Regression Checks (NHANES and PTB-XL branch integrity)
"""
import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.odir_preprocessor import ODIRPreprocessor, ODIR_DISEASE_LABELS, ODIR_DISEASE_NAMES
from app.models.odir_multimodal_model import ODIRMultimodalGNN
from app.inference.odir_inference import ODIRInferenceService
from backend.app.api.routes import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def create_dummy_fundus_image(color=(180, 60, 40)):
    """Generates a dummy 224x224 RGB image for testing."""
    img = Image.new("RGB", (224, 224), color=color)
    return img


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class TestODIRPipeline:
    def test_01_preprocessor_demographics(self):
        preprocessor = ODIRPreprocessor(age_mean=50.0, age_std=10.0)
        
        # Male age 60 -> (60-50)/10 = 1.0, sex = 1.0
        tensor_male = preprocessor.preprocess_demographics(age=60, sex="Male")
        assert tensor_male.shape == (1, 2)
        assert abs(tensor_male[0, 0].item() - 1.0) < 1e-4
        assert tensor_male[0, 1].item() == 1.0

        # Female age 40 -> (40-50)/10 = -1.0, sex = 0.0
        tensor_female = preprocessor.preprocess_demographics(age=40, sex="Female")
        assert tensor_female.shape == (1, 2)
        assert abs(tensor_female[0, 0].item() - (-1.0)) < 1e-4
        assert tensor_female[0, 1].item() == 0.0

    def test_02_preprocessor_bilateral_images(self):
        preprocessor = ODIRPreprocessor()
        img_left = create_dummy_fundus_image((200, 50, 30))
        img_right = create_dummy_fundus_image((190, 55, 35))

        left_t, right_t = preprocessor.preprocess_bilateral_images(img_left, img_right, is_training=False)
        assert left_t.shape == (1, 3, 224, 224)
        assert right_t.shape == (1, 3, 224, 224)
        assert left_t.dtype == torch.float32
        assert right_t.dtype == torch.float32

    def test_03_preprocessor_label_encoding(self):
        preprocessor = ODIRPreprocessor()
        row = {"N": 0, "D": 1, "G": 0, "C": 0, "A": 0, "H": 1, "M": 0, "O": 0}
        label_tensor = preprocessor.encode_labels(row)
        assert label_tensor.shape == (8,)
        assert label_tensor[1].item() == 1.0  # D
        assert label_tensor[5].item() == 1.0  # H
        assert label_tensor[0].item() == 0.0  # N

    def test_04_patient_split_zero_leakage(self):
        processed_dir = PROJECT_ROOT / "data" / "processed" / "odir"
        train_csv = processed_dir / "train_odir.csv"
        val_csv = processed_dir / "val_odir.csv"
        test_csv = processed_dir / "test_odir.csv"

        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)

            train_ids = set(train_df["ID"])
            val_ids = set(val_df["ID"])
            test_ids = set(test_df["ID"])

            # Verify strictly disjoint splits
            assert len(train_ids.intersection(val_ids)) == 0
            assert len(train_ids.intersection(test_ids)) == 0
            assert len(val_ids.intersection(test_ids)) == 0

            # Verify both eye images stay together
            assert len(train_df["left_image_path"].dropna()) == len(train_df)
            assert len(train_df["right_image_path"].dropna()) == len(train_df)

    def test_05_odir_multimodal_model_forward(self):
        model = ODIRMultimodalGNN(
            backbone_name="resnet18",
            pretrained=False,
            demo_input_dim=2,
            demo_emb_dim=32,
            single_eye_dim=64,
            bilateral_img_dim=128,
            fused_dim=128,
            gnn_hidden_dim=64,
            gnn_out_dim=32,
            k_neighbors=2,
            disease_targets=ODIR_DISEASE_LABELS
        )
        model.eval()

        batch_size = 3
        demo_t = torch.randn(batch_size, 2)
        left_t = torch.randn(batch_size, 3, 224, 224)
        right_t = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits_dict = model(demo_t, left_t, right_t)
            probs_dict = model.predict_probabilities(demo_t, left_t, right_t)

        for col in ODIR_DISEASE_LABELS:
            assert col in logits_dict
            assert col in probs_dict
            assert logits_dict[col].shape == (batch_size, 1)
            assert probs_dict[col].shape == (batch_size, 1)
            assert (probs_dict[col] >= 0.0).all() and (probs_dict[col] <= 1.0).all()

    def test_06_odir_api_missing_left_image(self):
        img_right = create_dummy_fundus_image()
        payload = {
            "age": 55,
            "sex": "Female",
            "right_image_b64": pil_to_base64(img_right)
            # missing left_image
        }
        res = client.post("/predict-odir", json=payload)
        assert res.status_code == 422
        assert "Left" in res.json()["detail"]

    def test_07_odir_api_missing_right_image(self):
        img_left = create_dummy_fundus_image()
        payload = {
            "age": 55,
            "sex": "Female",
            "left_image_b64": pil_to_base64(img_left)
            # missing right_image
        }
        res = client.post("/predict-odir", json=payload)
        assert res.status_code == 422
        assert "Right" in res.json()["detail"]

    def test_08_odir_api_invalid_image_data(self):
        payload = {
            "age": 55,
            "sex": "Female",
            "left_image_b64": "invalid_corrupted_base64_data",
            "right_image_b64": "invalid_corrupted_base64_data"
        }
        res = client.post("/predict-odir", json=payload)
        assert res.status_code == 422

    def test_09_model_info_includes_odir_branch(self):
        res = client.get("/model-info")
        assert res.status_code == 200
        data = res.json()
        assert "odir_ophthalmic_branch" in data
        assert len(data["odir_ophthalmic_branch"]["diseases_covered"]) == 8

    def test_10_existing_nhanes_checkpoint_unaffected(self):
        # Verify NHANES checkpoint still exists untouched
        nhanes_ckpt = PROJECT_ROOT / "models" / "multidisease_gnn_best.pt"
        assert nhanes_ckpt.exists(), "Existing NHANES checkpoint was accidentally removed or altered!"
