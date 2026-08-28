"""
Evaluation Service module.
Phase 11 — Computes comprehensive clinical model metrics (Accuracy, Precision, Recall,
Specificity, F1-Score, ROC-AUC, Confusion Matrix) and saves evaluation reports.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class EvaluationService:
    def __init__(self, output_dir: str = "outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold: float = 0.5,
        disease_name: str = "disease"
    ) -> Dict[str, Any]:
        """
        Evaluate binary disease prediction probabilities against ground truth.
        """
        y_pred_labels = (y_pred_probs >= threshold).astype(int)
        
        # Calculate confusion matrix components: TN, FP, FN, TP
        cm = confusion_matrix(y_true, y_pred_labels, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        acc = float(accuracy_score(y_true, y_pred_labels))
        prec = float(precision_score(y_true, y_pred_labels, zero_division=0))
        rec = float(recall_score(y_true, y_pred_labels, zero_division=0))
        f1 = float(f1_score(y_true, y_pred_labels, zero_division=0))
        
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        try:
            auc = float(roc_auc_score(y_true, y_pred_probs))
        except ValueError:
            auc = 0.5  # Default if single class in y_true
            
        metrics = {
            "disease": disease_name,
            "threshold": threshold,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }
        
        return metrics

    def save_report(self, metrics_dict: Dict[str, Any], filename: str = "evaluation_report.json"):
        """Save evaluation metrics to JSON and text summary."""
        file_path = self.output_dir / filename
        with open(file_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
            
        txt_path = self.output_dir / filename.replace(".json", ".txt")
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CLINICAL DECISION SUPPORT EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            for disease, m in metrics_dict.items():
                if isinstance(m, dict) and "accuracy" in m:
                    f.write(f"🩺 Condition: {disease.upper()}\n")
                    f.write(f"   • Accuracy:          {m['accuracy'] * 100:.2f}%\n")
                    f.write(f"   • Precision:         {m['precision'] * 100:.2f}%\n")
                    f.write(f"   • Recall/Sensitivity:{m['recall_sensitivity'] * 100:.2f}%\n")
                    f.write(f"   • Specificity:       {m['specificity'] * 100:.2f}%\n")
                    f.write(f"   • F1-Score:          {m['f1_score']:.4f}\n")
                    f.write(f"   • ROC-AUC:           {m['roc_auc']:.4f}\n")
                    f.write(f"   • Confusion Matrix:  {m['confusion_matrix']}\n\n")
                    
        return file_path
