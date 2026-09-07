"""
==============================================================================
PTB-XL Multimodal GNN Training Pipeline (Phase 5)
==============================================================================
Trains a multimodal heart-disease model combining:
- PTB-XL Clinical/Demographic features (Age, Sex, Height, Weight)
- PTB-XL 12-lead ECG signals (100Hz, 10-second waveforms)
- Multimodal Patient Representation Fusion
- Patient Similarity Graph Construction (Cosine Similarity k-NN)
- Graph Neural Network (GNN) Message Passing Backbone
- Binary Classification: Normal ECG (0) vs Abnormal ECG (1)

Strictly leak-free: Evaluated exclusively on patient-level partitions.
==============================================================================
"""

import os
import sys
import random
import argparse
import time
# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Project path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from app.models.clinical_encoder import ClinicalEncoder
from app.models.ecg_encoder import ECGEncoder
from app.models.multimodal_model import MultimodalFusion
from app.models.gnn_model import MultiDiseaseGNN
from app.graph.patient_graph import PatientGraphBuilder


def set_seed(seed: int = 42):
    """Sets random seeds for full reproducibility across NumPy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Dataset Class
# ----------------------------------------------------------------------
class PTBXLDataset(Dataset):
    """Loads paired tabular clinical features, 12-lead ECG waveforms, and binary targets."""
    def __init__(self, meta_csv_path: str, preprocessor: PTBXLPreprocessor, max_samples: Optional[int] = None):
        self.df = pd.read_csv(meta_csv_path)
        if max_samples and max_samples < len(self.df):
            self.df = self.df.iloc[:max_samples].copy()
            
        self.preprocessor = preprocessor
        self.clinical_cols = preprocessor.clinical_cols
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Clinical tabular features
        clinical_vals = row[self.clinical_cols].values.astype(np.float32)
        clinical = torch.tensor(clinical_vals, dtype=torch.float32)
        
        # Binary target
        target = torch.tensor(row['target'], dtype=torch.float32)
        
        # 12-lead ECG waveform: shape (12, 1000)
        filename_lr = row['filename_lr']
        ecg_signal = self.preprocessor.extract_waveform(filename_lr)
        ecg_sig_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
        
        return clinical, ecg_sig_tensor, target


# ----------------------------------------------------------------------
# End-to-End Multimodal GNN Model
# ----------------------------------------------------------------------
class PTBXLMultimodalGNN(nn.Module):
    """
    End-to-End Multimodal Architecture:
    ClinicalEncoder (64-dim) + ECG 1D CNN Encoder (128-dim)
      -> Multimodal Fusion (128-dim)
      -> Patient Similarity Graph Builder
      -> MultiDiseaseGNN Backbone
      -> Binary Prediction Logits
    """
    def __init__(
        self,
        clinical_in: int = 4,
        ecg_channels: int = 12,
        clinical_emb_dim: int = 64,
        ecg_emb_dim: int = 128,
        fused_dim: int = 128,
        gnn_hidden: int = 64,
        gnn_out: int = 32,
        k_neighbors: int = 5,
        dropout_rate: float = 0.2
    ):
        super(PTBXLMultimodalGNN, self).__init__()
        self.clinical_enc = ClinicalEncoder(
            input_dim=clinical_in,
            embedding_dim=clinical_emb_dim,
            hidden_dims=[128, 64],
            dropout_rate=dropout_rate
        )
        self.ecg_enc = ECGEncoder(in_channels=ecg_channels, out_dim=ecg_emb_dim)
        self.fusion = MultimodalFusion(
            clinical_dim=clinical_emb_dim,
            image_dim=ecg_emb_dim,
            fused_dim=fused_dim,
            dropout_rate=dropout_rate
        )
        self.graph_builder = PatientGraphBuilder(k=k_neighbors)
        self.gnn = MultiDiseaseGNN(
            in_channels=fused_dim,
            hidden_channels=gnn_hidden,
            out_channels=gnn_out,
            disease_targets=["heart_disease"],
            dropout_rate=dropout_rate
        )
        
    def forward(
        self,
        clinical_x: torch.Tensor,
        ecg_x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a batch/cluster of patient recordings.
        """
        # 1. Modality Encodings
        c_emb = self.clinical_enc(clinical_x)
        e_emb = self.ecg_enc(ecg_x)
        
        # 2. Multimodal Fusion
        f_emb = self.fusion(c_emb, e_emb, is_linked_patient=True)
        
        # 3. Dynamic Patient Graph Construction (if not pre-supplied)
        if edge_index is None:
            edge_index, _, _ = self.graph_builder.build_graph(f_emb)
            
        # 4. GNN Message Passing & Disease Head Prediction
        gnn_out = self.gnn(f_emb, edge_index)
        logits = gnn_out["heart_disease"].squeeze(-1)
        return logits


# ----------------------------------------------------------------------
# Training Routine
# ----------------------------------------------------------------------
def train_model(
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    is_smoke_test: bool = False
) -> Dict[str, Any]:
    print("=" * 70)
    print("🚀 STARTING PTB-XL MULTIMODAL GNN TRAINING")
    print("=" * 70)
    
    # 1. Reproducibility & Hardware Detection
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Hardware Device: {device}")
    if device.type == 'cpu':
        print("⚠️ Running on CPU (CUDA not detected/required).")
    print(f"🌱 Random Seed: {seed}")
        
    # 2. Resolve Directory Paths
    data_dir = PROJECT_ROOT / "data" / "ptbxl"
    models_dir = PROJECT_ROOT / "models" / "ptbxl_multimodal"
    reports_dir = PROJECT_ROOT / "reports"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    extract_path = data_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if not extract_path.exists():
        extract_path = data_dir
        
    preprocessor = PTBXLPreprocessor(data_dir=str(extract_path))
    
    # Check or generate patient-level splits
    train_csv = data_dir / "train_metadata.csv"
    val_csv = data_dir / "val_metadata.csv"
    test_csv = data_dir / "test_metadata.csv"
    
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        print("Preprocessing and generating patient-level splits...")
        preprocessor.process_data()
        
    # 3. Create Datasets & DataLoaders
    train_ds = PTBXLDataset(str(train_csv), preprocessor, max_samples=max_train_samples)
    val_ds = PTBXLDataset(str(val_csv), preprocessor, max_samples=max_val_samples)
    
    # Read test info for split report without evaluating it during training
    df_test = pd.read_csv(test_csv)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Check patient split integrity
    train_pts = set(df_train['patient_id'])
    val_pts = set(df_val['patient_id'])
    test_pts = set(df_test['patient_id'])
    
    train_val_overlap = len(train_pts.intersection(val_pts))
    train_test_overlap = len(train_pts.intersection(test_pts))
    val_test_overlap = len(val_pts.intersection(test_pts))
    
    leakage_passed = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)
    assert leakage_passed, "Critical error: Patient leakage detected across splits!"
    
    num_train_patients = len(train_pts)
    num_val_patients = len(val_pts)
    num_test_patients = len(test_pts)
    
    train_targets_all = df_train['target'].values
    train_pos_all = int((train_targets_all == 1).sum())
    train_neg_all = int((train_targets_all == 0).sum())
    
    print(f"📊 Dataset Partitions (Patient-Level):")
    print(f"   • Train: {num_train_patients} patients ({len(train_ds)} / {len(df_train)} records used)")
    print(f"   • Val:   {num_val_patients} patients ({len(val_ds)} / {len(df_val)} records used)")
    print(f"   • Test:  {num_test_patients} patients ({len(df_test)} records untouched)")
    print(f"   • Patient Overlap: Train-Val={train_val_overlap}, Train-Test={train_test_overlap}, Val-Test={val_test_overlap} (PASS)")
    
    # 4. Model, Criterion, Optimizer
    model = PTBXLMultimodalGNN(
        clinical_in=4,
        ecg_channels=12,
        clinical_emb_dim=64,
        ecg_emb_dim=128,
        fused_dim=128,
        gnn_hidden=64,
        gnn_out=32,
        k_neighbors=5,
        dropout_rate=0.2
    ).to(device)
    
    # Calculate positive weight for class balance based on used training subset
    train_targets_used = train_ds.df['target'].values
    pos_count = (train_targets_used == 1).sum()
    neg_count = (train_targets_used == 0).sum()
    pos_weight_val = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 5. Training Loop with Early Stopping
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_prec = 0.0
    best_val_rec = 0.0
    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    best_checkpoint_path = models_dir / "best_model.pt"
    classifier_checkpoint_path = models_dir / "ptbxl_classifier.pt"
    ecg_encoder_path = models_dir / "ecg_encoder.pt"
    ecg_encoder_alt_path = models_dir / "ptbxl_ecg_encoder.pt"
    
    history: List[Dict[str, Any]] = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        
        for batch_idx, (c_x, e_x, y) in enumerate(train_loader):
            c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(c_x, e_x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss_total / max(train_batches, 1)
        
        # Validation Pass (Inductive Batch Evaluation on Validation Split ONLY)
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_val_logits: List[float] = []
        all_val_targets: List[int] = []
        
        with torch.no_grad():
            for c_x, e_x, y in val_loader:
                c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
                logits = model(c_x, e_x)
                loss = criterion(logits, y)
                
                val_loss_total += loss.item()
                val_batches += 1
                
                all_val_logits.extend(logits.cpu().numpy().tolist())
                all_val_targets.extend(y.cpu().numpy().astype(int).tolist())
                
        avg_val_loss = val_loss_total / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        val_probs = 1.0 / (1.0 + np.exp(-np.array(all_val_logits)))
        val_preds = (val_probs >= 0.5).astype(int)
        y_val_true = np.array(all_val_targets)
        
        val_acc = float(accuracy_score(y_val_true, val_preds))
        val_prec = float(precision_score(y_val_true, val_preds, zero_division=0))
        val_rec = float(recall_score(y_val_true, val_preds, zero_division=0))
        val_f1 = float(f1_score(y_val_true, val_preds, zero_division=0))
        try:
            val_auc = float(roc_auc_score(y_val_true, val_probs))
        except Exception:
            val_auc = 0.5
            
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Prec: {val_prec:.4f} | "
              f"Val Rec: {val_rec:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val AUC: {val_auc:.4f}")
              
        history.append({
            'epoch': epoch,
            'train_loss': round(float(avg_train_loss), 4),
            'val_loss': round(float(avg_val_loss), 4),
            'val_accuracy': round(val_acc, 4),
            'val_precision': round(val_prec, 4),
            'val_recall': round(val_rec, 4),
            'val_f1': round(val_f1, 4),
            'val_roc_auc': round(val_auc, 4),
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Checkpoint Best Model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_prec = val_prec
            best_val_rec = val_rec
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save checkpoints
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_acc,
                'best_val_precision': best_val_prec,
                'best_val_recall': best_val_rec,
                'best_val_f1': best_val_f1,
                'best_val_roc_auc': best_val_auc,
                'architecture': 'PTBXLMultimodalGNN',
                'config': {
                    'clinical_in': 4,
                    'ecg_channels': 12,
                    'clinical_emb_dim': 64,
                    'ecg_emb_dim': 128,
                    'fused_dim': 128,
                    'gnn_hidden': 64,
                    'k_neighbors': 5
                }
            }, best_checkpoint_path)
            
            torch.save(model.state_dict(), classifier_checkpoint_path)
            torch.save(model.ecg_enc.state_dict(), ecg_encoder_path)
            torch.save(model.ecg_enc.state_dict(), ecg_encoder_alt_path)
            print(f"  ⭐ New best checkpoint saved at epoch {epoch} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs (no improvement in {patience} epochs).")
                break
                
    total_train_time = time.time() - start_time
    print("=" * 70)
    print(f"🎉 Training Finished in {total_train_time:.1f}s. Best Epoch: {best_epoch}")
    print("=" * 70)
    
    # 6. Save Metrics CSV
    df_metrics = pd.DataFrame(history)
    metrics_csv_path = reports_dir / "training_metrics.csv"
    alt_metrics_csv_path = reports_dir / "ptbxl_training_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    df_metrics.to_csv(alt_metrics_csv_path, index=False)
    
    # Generate Training Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_metrics['epoch'], df_metrics['train_loss'], 'o-', label='Train Loss', color='#2563eb')
    plt.plot(df_metrics['epoch'], df_metrics['val_loss'], 's--', label='Val Loss', color='#dc2626')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df_metrics['epoch'], df_metrics['val_accuracy'], 'o-', label='Val Accuracy', color='#16a34a')
    plt.plot(df_metrics['epoch'], df_metrics['val_f1'], '^--', label='Val F1', color='#9333ea')
    plt.plot(df_metrics['epoch'], df_metrics['val_roc_auc'], 'd:', label='Val ROC-AUC', color='#ea580c')
    plt.title('Validation Performance Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    curves_path = reports_dir / "ptbxl_training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    
    # Write Training Report Markdown
    report_content = f"""# PTB-XL Multimodal GNN Training Report

**Model Architecture**: `PTBXLMultimodalGNN` (`ClinicalEncoder` + `ECGEncoder` + `MultimodalFusion` + `PatientGraphBuilder` + `MultiDiseaseGNN`)  
**Execution Mode**: {'Smoke Test Run' if is_smoke_test else 'Full Training Run'}  
**Hardware Device**: {device.type.upper()}  
**Random Seed**: {seed}  
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Dataset & Patient-Level Splitting
- **Total Dataset Records**: {len(df_train) + len(df_val) + len(df_test)}
- **Total Unique Patients**: {num_train_patients + num_val_patients + num_test_patients}
- **Training Set**: {num_train_patients} unique patients ({len(train_ds)} / {len(df_train)} records used)
- **Validation Set**: {num_val_patients} unique patients ({len(val_ds)} / {len(df_val)} records used)
- **Test Set (Untouched)**: {num_test_patients} unique patients ({len(df_test)} records reserved)
- **Class Distribution (Train)**: {train_neg_all} Normal ({train_neg_all/len(df_train)*100:.1f}%) / {train_pos_all} Abnormal ({train_pos_all/len(df_train)*100:.1f}%)
- **Split Integrity Verification**:
  - Train-Val Patient Overlap: {train_val_overlap} (Zero Leakage)
  - Train-Test Patient Overlap: {train_test_overlap} (Zero Leakage)
  - Val-Test Patient Overlap: {val_test_overlap} (Zero Leakage)
  - Split Status: **PASS (Patient-Disjoint Partitions Confirmed)**

---

## 2. Model Architecture & Hyperparameters
- **Clinical Feature Inputs**: 4 features (Age, Sex, Height, Weight) -> `ClinicalEncoder` (64-dim)
- **ECG Waveform Inputs**: 12 leads x 1000 timepoints (100Hz 10s) -> `ECGEncoder` 1D CNN (128-dim)
- **Multimodal Fusion**: Cross-modality projection -> `MultimodalFusion` (128-dim)
- **Graph Topology**: `PatientGraphBuilder` (k={5} nearest neighbors via Cosine Similarity)
- **GNN Backbone**: `MultiDiseaseGNN` (64 hidden -> 32 output -> binary disease logits)
- **Loss Function**: `BCEWithLogitsLoss` (pos_weight={pos_weight_val:.2f})
- **Optimizer**: `AdamW` (learning_rate={learning_rate}, weight_decay={weight_decay})
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2)
- **Batch Size**: {batch_size}
- **Early Stopping Patience**: {patience} epochs

---

## 3. Best Validation Performance (Epoch {best_epoch})
- **Best Validation Loss**: {best_val_loss:.4f}
- **Validation Accuracy**: {best_val_acc * 100:.2f}%
- **Validation Precision**: {best_val_prec:.4f}
- **Validation Recall / Sensitivity**: {best_val_rec:.4f}
- **Validation F1-Score**: {best_val_f1:.4f}
- **Validation ROC-AUC**: {best_val_auc:.4f}

---

## 4. Epoch Metrics Summary

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | LR |
|---|---|---|---|---|---|---|---|---|
"""
    for row in history:
        report_content += f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_accuracy']:.4f} | {row['val_precision']:.4f} | {row['val_recall']:.4f} | {row['val_f1']:.4f} | {row['val_roc_auc']:.4f} | {row['learning_rate']:.1e} |\n"
        
    report_content += f"""
---

## 5. Generated Artifacts & Checkpoints
- **Best Multimodal Model Checkpoint**: `models/ptbxl_multimodal/best_model.pt`
- **ECG Encoder Checkpoint**: `models/ptbxl_multimodal/ecg_encoder.pt`
- **Training Metrics CSV**: `reports/training_metrics.csv`
- **Training Curves Plot**: `reports/ptbxl_training_curves.png`
"""
    report_md_path = reports_dir / "PTBXL_TRAINING_REPORT.md"
    alt_report_md_path = reports_dir / "ptbxl_training_report.md"
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    with open(alt_report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"📄 Saved training metrics to: {metrics_csv_path}")
    print(f"📊 Saved loss curves to: {curves_path}")
    print(f"📝 Saved training report to: {report_md_path}")
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_prec': best_val_prec,
        'best_val_rec': best_val_rec,
        'best_val_f1': best_val_f1,
        'best_val_auc': best_val_auc,
        'checkpoint_path': str(best_checkpoint_path),
        'metrics_csv': str(metrics_csv_path),
        'report_path': str(report_md_path),
        'history': history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PTB-XL Multimodal GNN")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for train/val subsets (for smoke testing)")
    parser.add_argument("--max-train", type=int, default=None, help="Max training samples")
    parser.add_argument("--max-val", type=int, default=None, help="Max val samples")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num workers (default 0 for CPU/Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smoke-test", action="store_true", help="Execute fast smoke test")
    
    args = parser.parse_args()
    
    max_train = args.max_train or args.max_samples
    max_val = args.max_val or (args.max_samples // 2 if args.max_samples else None)
    
    if args.smoke_test:
        train_model(
            epochs=args.epochs if args.epochs != 5 else 2,
            batch_size=args.batch_size if args.batch_size != 64 else 16,
            max_train_samples=max_train or 100,
            max_val_samples=max_val or 50,
            num_workers=args.num_workers,
            seed=args.seed,
            is_smoke_test=True
        )
    else:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            max_train_samples=max_train,
            max_val_samples=max_val,
            num_workers=args.num_workers,
            seed=args.seed
        )
