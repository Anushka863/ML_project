"""
Tabular Feature Explainability (XAI) module.
Phase 12 — Computes model feature attributions (SHAP / Feature Permutation Importance)
for clinical decision support explanations.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Optional


class TabularXAI:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def compute_feature_attributions(
        self,
        model: nn.Module,
        input_vector: np.ndarray,
        baseline_vector: Optional[np.ndarray] = None,
        target_disease: str = "diabetes"
    ) -> List[Dict[str, Any]]:
        """
        Computes feature attribution scores for a single patient input using gradient x input.
        Returns a sorted list of features with importance scores and directional impact.
        """
        model.eval()
        
        if input_vector.ndim == 1:
            input_vector = input_vector.reshape(1, -1)
            
        x_tensor = torch.tensor(input_vector, dtype=torch.float32, requires_grad=True)
        
        # Forward pass depending on model signature
        if hasattr(model, "forward") and callable(model.forward):
            try:
                out = model(x_tensor)
            except TypeError:
                # If model expects edge_index
                dummy_edge = torch.empty((2, 0), dtype=torch.long)
                out = model(x_tensor, dummy_edge)
        else:
            out = model(x_tensor)
            
        if isinstance(out, dict):
            logit = out[target_disease]
        else:
            logit = out
            
        score = torch.sigmoid(logit).sum()
        score.backward()
        
        grads = x_tensor.grad.detach().cpu().numpy().flatten()
        vals = input_vector.flatten()
        
        # Attribution = gradient * value
        attributions = grads * vals
        
        results = []
        for i, fname in enumerate(self.feature_names):
            if i < len(attributions):
                imp = float(attributions[i])
                direction = "increases risk" if imp > 0 else "lowers risk"
                results.append({
                    "feature": fname,
                    "value": float(vals[i]),
                    "importance": round(imp, 4),
                    "impact": direction
                })
                
        # Sort by absolute importance descending
        results.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return results
