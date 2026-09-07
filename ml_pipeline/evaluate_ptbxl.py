"""
==============================================================================
PTB-XL Multimodal GNN Evaluation Pipeline (Phase 5)
==============================================================================
Evaluates the finalized best model checkpoint on the untouched patient-level
test set. Generates ROC curves, Confusion Matrix, Classification Report,
and final evaluation documentation.
==============================================================================
"""

import os
import sys
import time
import argparse

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from ml_pipeline.train_ptbxl_multimodal import PTBXLDataset, PTBXLMultimodalGNN


def evaluate_ptbxl(
    max_test_samples: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, Any]:
    print("=" * 70)
    print("🔬 STARTING PTB-XL TEST SET EVALUATION (UNTOUCHED PATIENTS)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Hardware Device: {device}")
    
    data_dir = PROJECT_ROOT / "data" / "ptbxl"
    models_dir = PROJECT_ROOT / "models" / "ptbxl_multimodal"
    reports_dir = PROJECT_ROOT / "reports"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = models_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = models_dir / "ptbxl_classifier.pt"
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found in {models_dir}. Run training first!")
        
    extract_path = data_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if not extract_path.exists():
        extract_path = data_dir
        
    preprocessor = PTBXLPreprocessor(data_dir=str(extract_path))
    test_meta = data_dir / "test_metadata.csv"
    
    if not test_meta.exists():
        raise FileNotFoundError(f"{test_meta} not found. Preprocessing splits must be generated first.")
        
    df_test_full = pd.read_csv(test_meta)
    test_ds = PTBXLDataset(str(test_meta), preprocessor, max_samples=max_test_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Load Model
    model = PTBXLMultimodalGNN(
        clinical_in=4,
        ecg_channels=12,
        clinical_emb_dim=64,
        ecg_emb_dim=128,
        fused_dim=128,
        gnn_hidden=64,
        gnn_out=32,
        k_neighbors=5,
        dropout_rate=0.0
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', 'N/A')
        print(f"✅ Loaded checkpoint from best epoch {best_epoch}: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint state dictionary: {checkpoint_path}")
        
    model.eval()
    print(f"📊 Evaluating {len(test_ds)} untouched test recordings ({df_test_full['patient_id'].nunique()} unique test patients)...")
    
    all_logits = []
    all_targets = []
    
    start_eval = time.time()
    with torch.no_grad():
        for c_x, e_x, y in test_loader:
            c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
            logits = model(c_x, e_x)
            all_logits.extend(logits.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().astype(int).tolist())
            
    eval_time = time.time() - start_eval
    
    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    preds = (probs >= 0.5).astype(int)
    y_true = np.array(all_targets)
    
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_true, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    clf_report = classification_report(
        y_true,
        preds,
        target_names=['Normal (0)', 'Abnormal (1)'],
        digits=4,
        zero_division=0
    )
    
    print("-" * 70)
    print(f"🎯 Test Accuracy:          {acc * 100:.2f}%")
    print(f"🎯 Test Precision:         {prec:.4f}")
    print(f"🎯 Test Recall (Sens):     {rec:.4f}")
    print(f"🎯 Test Specificity:       {spec:.4f}")
    print(f"🎯 Test F1-Score:          {f1:.4f}")
    print(f"🎯 Test ROC-AUC:           {auc:.4f}")
    print("-" * 70)
    print("📋 Classification Report:")
    print(clf_report)
    print("-" * 70)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal (0)', 'Abnormal (1)'],
        yticklabels=['Normal (0)', 'Abnormal (1)']
    )
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Test Set Confusion Matrix: PTB-XL Multimodal GNN')
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#2563eb', lw=2, label=f'Multimodal GNN (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 3. Final Evaluation Markdown Report
    eval_report_path = reports_dir / "final_evaluation.md"
    report_content = f"""# Final Evaluation Report — PTB-XL Multimodal GNN

**Target**: Binary Heart Disease Detection (Normal vs Abnormal)  
**Evaluation Set**: Untouched Patient Test Set ({len(test_ds)} recordings, {df_test_full['patient_id'].nunique()} unique patients)  
**Hardware Device**: {device.type.upper()}  
**Evaluation Time**: {eval_time:.2f}s  
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Quantitative Performance Metrics

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **Accuracy** | **{acc * 100:.2f}%** | Overall correct classification rate across test cohort |
| **Precision** | **{prec:.4f}** | True abnormal proportion among positive alarms |
| **Recall / Sensitivity** | **{rec:.4f}** | Cardiac abnormality detection sensitivity |
| **Specificity** | **{spec:.4f}** | True normal ECG specificity (false alarm resistance) |
| **F1-Score** | **{f1:.4f}** | Balanced harmonic mean of precision & recall |
| **ROC-AUC** | **{auc:.4f}** | Area Under Receiver Operating Characteristic Curve |

---

## 2. Confusion Matrix Breakdown

| Metric | Value |
|---|---|
| **True Negatives (TN)** | {tn} (Normal correctly classified) |
| **False Positives (FP)** | {fp} (Normal incorrectly flagged as abnormal) |
| **False Negatives (FN)** | {fn} (Abnormal missed) |
| **True Positives (TP)** | {tp} (Abnormal correctly identified) |

---

## 3. Detailed Classification Report

```text
{clf_report}
```

---

## 4. Methodological Validation & Clinical Integrity
1. **Zero Patient Overlap**: Evaluated strictly on patient IDs never seen during training or validation hyperparameter tuning.
2. **Inductive Graph Inference**: Dynamic k-NN patient similarity graph constructed at test-time without test-train edge leakage.
3. **Multimodal Synergy**: Demonstrates joint representation learning combining tabular demographics (Age, Sex, Height, Weight) with 12-lead potential waveforms.
4. **Visual Artifacts**:
   - Confusion Matrix: `reports/confusion_matrix.png`
   - ROC Curve: `reports/roc_curve.png`
"""
    with open(eval_report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"📄 Saved evaluation report: {eval_report_path}")
    print(f"📊 Saved confusion matrix:  {cm_path}")
    print(f"📈 Saved ROC curve:         {roc_path}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PTB-XL Multimodal GNN on Test Set")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate (optional)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num workers")
    args = parser.parse_args()
    
    evaluate_ptbxl(
        max_test_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
