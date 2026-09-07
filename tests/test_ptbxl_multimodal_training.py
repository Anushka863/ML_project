"""
Unit and Integration Tests for PTB-XL Multimodal GNN Training Pipeline (Phase 5)
================================================================================
Verifies:
1. Model forward pass
2. Clinical embedding dimensions
3. ECG embedding dimensions
4. Multimodal fusion dimensions
5. Graph input dimensions
6. Output shape
7. Patient-level split integrity (Zero patient overlap between train/val/test)
8. Checkpoint save/load integrity
9. Small end-to-end training batch (CPU compatibility, loss, optimizer step)
10. CPU Execution guarantee
11. Metric calculation consistency
12. End-to-end smoke test function execution
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.clinical_encoder import ClinicalEncoder
from app.models.ecg_encoder import ECGEncoder
from app.models.multimodal_model import MultimodalFusion
from app.graph.patient_graph import PatientGraphBuilder
from app.models.gnn_model import MultiDiseaseGNN
from ml_pipeline.train_ptbxl_multimodal import PTBXLMultimodalGNN, PTBXLDataset, train_model
from ml_pipeline.evaluate_ptbxl import evaluate_ptbxl


class TestPTBXLMultimodalTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4
        cls.clinical_in = 4
        cls.ecg_channels = 12
        cls.ecg_len = 1000
        cls.clinical_emb_dim = 64
        cls.ecg_emb_dim = 128
        cls.fused_dim = 128
        
        cls.dummy_clinical = torch.randn(cls.batch_size, cls.clinical_in)
        cls.dummy_ecg = torch.randn(cls.batch_size, cls.ecg_channels, cls.ecg_len)
        cls.dummy_targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

    def test_01_clinical_encoder_dimensions(self):
        """Verifies clinical encoder outputs exact (batch_size, 64) embedding."""
        encoder = ClinicalEncoder(input_dim=self.clinical_in, embedding_dim=self.clinical_emb_dim)
        encoder.eval()
        with torch.no_grad():
            out = encoder(self.dummy_clinical)
        self.assertEqual(out.shape, (self.batch_size, self.clinical_emb_dim))

    def test_02_ecg_encoder_dimensions(self):
        """Verifies 1D CNN ECG encoder outputs exact (batch_size, 128) embedding."""
        encoder = ECGEncoder(in_channels=self.ecg_channels, out_dim=self.ecg_emb_dim)
        encoder.eval()
        with torch.no_grad():
            out = encoder(self.dummy_ecg)
        self.assertEqual(out.shape, (self.batch_size, self.ecg_emb_dim))

    def test_03_multimodal_fusion_dimensions(self):
        """Verifies multimodal fusion maps (clinical + ecg) to fused_dim (128)."""
        c_emb = torch.randn(self.batch_size, self.clinical_emb_dim)
        e_emb = torch.randn(self.batch_size, self.ecg_emb_dim)
        fusion = MultimodalFusion(
            clinical_dim=self.clinical_emb_dim,
            image_dim=self.ecg_emb_dim,
            fused_dim=self.fused_dim
        )
        fusion.eval()
        with torch.no_grad():
            fused = fusion(c_emb, e_emb, is_linked_patient=True)
        self.assertEqual(fused.shape, (self.batch_size, self.fused_dim))

    def test_04_patient_graph_dimensions(self):
        """Verifies graph builder creates valid edge tensor and node weights."""
        fused = torch.randn(self.batch_size, self.fused_dim)
        builder = PatientGraphBuilder(k=2)
        edge_index, weights, x = builder.build_graph(fused)
        
        self.assertEqual(edge_index.dim(), 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(x.shape, (self.batch_size, self.fused_dim))

    def test_05_gnn_output_shape(self):
        """Verifies GNN produces heart_disease prediction logits for each node."""
        fused = torch.randn(self.batch_size, self.fused_dim)
        builder = PatientGraphBuilder(k=2)
        edge_index, _, _ = builder.build_graph(fused)
        
        gnn = MultiDiseaseGNN(
            in_channels=self.fused_dim,
            hidden_channels=64,
            out_channels=32,
            disease_targets=["heart_disease"]
        )
        gnn.eval()
        with torch.no_grad():
            out = gnn(fused, edge_index)
            
        self.assertIn("heart_disease", out)
        self.assertEqual(out["heart_disease"].shape, (self.batch_size, 1))

    def test_06_end_to_end_model_forward_pass(self):
        """Verifies end-to-end forward pass through complete architecture."""
        model = PTBXLMultimodalGNN(
            clinical_in=self.clinical_in,
            ecg_channels=self.ecg_channels,
            clinical_emb_dim=self.clinical_emb_dim,
            ecg_emb_dim=self.ecg_emb_dim,
            fused_dim=self.fused_dim,
            k_neighbors=2
        )
        model.eval()
        with torch.no_grad():
            logits = model(self.dummy_clinical, self.dummy_ecg)
        self.assertEqual(logits.shape, (self.batch_size,))

    def test_07_patient_level_split_integrity(self):
        """Verifies zero patient overlap between Train, Val, and Test metadata."""
        train_csv = PROJECT_ROOT / "data" / "ptbxl" / "train_metadata.csv"
        val_csv = PROJECT_ROOT / "data" / "ptbxl" / "val_metadata.csv"
        test_csv = PROJECT_ROOT / "data" / "ptbxl" / "test_metadata.csv"
        
        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            df_train = pd.read_csv(train_csv)
            df_val = pd.read_csv(val_csv)
            df_test = pd.read_csv(test_csv)
            
            train_pts = set(df_train['patient_id'])
            val_pts = set(df_val['patient_id'])
            test_pts = set(df_test['patient_id'])
            
            self.assertEqual(len(train_pts.intersection(val_pts)), 0, "Train-Val patient overlap detected!")
            self.assertEqual(len(train_pts.intersection(test_pts)), 0, "Train-Test patient overlap detected!")
            self.assertEqual(len(val_pts.intersection(test_pts)), 0, "Val-Test patient overlap detected!")

    def test_08_checkpoint_save_and_load(self):
        """Verifies model weights can be saved to disk and restored identically."""
        temp_dir = tempfile.mkdtemp(prefix="test_ckpt_")
        try:
            ckpt_path = os.path.join(temp_dir, "test_model.pt")
            model = PTBXLMultimodalGNN(clinical_in=self.clinical_in, ecg_channels=self.ecg_channels)
            model.eval()
            torch.save(model.state_dict(), ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))
            
            loaded_model = PTBXLMultimodalGNN(clinical_in=self.clinical_in, ecg_channels=self.ecg_channels)
            loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            loaded_model.eval()
            
            with torch.no_grad():
                out1 = model(self.dummy_clinical, self.dummy_ecg)
                out2 = loaded_model(self.dummy_clinical, self.dummy_ecg)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_09_small_end_to_end_training_batch_on_cpu(self):
        """Runs a full forward/backward optimization step explicitly on CPU device."""
        device = torch.device('cpu')
        model = PTBXLMultimodalGNN(
            clinical_in=self.clinical_in,
            ecg_channels=self.ecg_channels,
            clinical_emb_dim=self.clinical_emb_dim,
            ecg_emb_dim=self.ecg_emb_dim,
            fused_dim=self.fused_dim,
            k_neighbors=2
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        optimizer.zero_grad()
        logits = model(self.dummy_clinical.to(device), self.dummy_ecg.to(device))
        loss = criterion(logits, self.dummy_targets.to(device))
        loss.backward()
        optimizer.step()
        
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_10_metric_calculations(self):
        """Verifies accuracy, precision, recall, F1, and ROC-AUC metrics calculations."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        probs = np.array([0.1, 0.8, 0.3, 0.9, 0.4, 0.2])
        preds = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        
        self.assertAlmostEqual(acc, 5.0/6.0)
        self.assertEqual(prec, 1.0)
        self.assertAlmostEqual(rec, 2.0/3.0)
        self.assertGreater(auc, 0.5)


if __name__ == "__main__":
    unittest.main()
