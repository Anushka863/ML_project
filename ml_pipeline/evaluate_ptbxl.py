import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from ml_pipeline.train_ptbxl_multimodal import PTBXLDataset, PTBXLClassifier

def evaluate_ptbxl():
    print("==================================================")
    print("PHASE 10 — EVALUATION PIPELINE (PTB-XL)")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r"c:\Users\Aisha Fathima\OneDrive\Desktop\ML Project\ML_project"
    data_dir = os.path.join(base_dir, "data", "ptbxl")
    reports_dir = os.path.join(base_dir, "reports")
    model_path = os.path.join(base_dir, "models", "ptbxl_multimodal", "ptbxl_classifier.pt")
    
    extract_path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    if not os.path.exists(extract_path):
         extract_path = data_dir
         
    preprocessor = PTBXLPreprocessor(data_dir=extract_path)
    
    test_meta = os.path.join(data_dir, "test_metadata.csv")
    if not os.path.exists(test_meta):
        print(f"Error: {test_meta} not found. Run training/preprocessing first.")
        return
        
    test_ds = PTBXLDataset(test_meta, preprocessor)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = PTBXLClassifier(clinical_in=4, ecg_channels=12)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_probs = []
    all_preds = []
    all_y = []
    
    print("Evaluating on untouched test set...")
    with torch.no_grad():
        for c_x, e_x, y in test_loader:
            c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
            logits = model(c_x, e_x)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().int().numpy())
            all_y.extend(y.cpu().int().numpy())
            
    # Metrics
    y_true = np.array(all_y)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: PTB-XL Heart PoC')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "roc_curve.png"))
    plt.close()
    
    # 3. Final Evaluation MD
    lines = [
        "# Final Evaluation Report - PTB-XL Multimodal PoC",
        "",
        "## Metrics on Untouched Patient Test Set",
        f"- **Accuracy**: {acc:.4f}",
        f"- **Precision**: {prec:.4f}",
        f"- **Recall/Sensitivity**: {rec:.4f}",
        f"- **Specificity**: {spec:.4f}",
        f"- **F1-Score**: {f1:.4f}",
        f"- **ROC-AUC**: {auc:.4f}",
        "",
        "## Notes",
        "- The model fuses age, sex, weight, and height along with the 12-lead structural ECG signals.",
        "- Evaluated strictly grouped by patient ID to prevent temporal signal leakage."
    ]
    
    with open(os.path.join(reports_dir, "final_evaluation.md"), "w") as f:
        f.write("\n".join(lines))
        
    print("Evaluation completed. Saved reports/confusion_matrix.png, reports/roc_curve.png, reports/final_evaluation.md")

if __name__ == "__main__":
    evaluate_ptbxl()
