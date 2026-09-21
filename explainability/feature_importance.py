"""
==============================================================================
Feature Importance Module for PTB-XL Multimodal GNN
==============================================================================
Derives genuine model feature attributions for clinical demographic inputs
(age, sex, height, weight) using Integrated Gradients.
==============================================================================
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from explainability.ptbxl_explainer import PTBXLExplainer, PTBXL_CLINICAL_COLS


def get_clinical_feature_attributions(
    model: nn.Module,
    clinical_tensor: torch.Tensor,
    ecg_tensor: torch.Tensor,
    raw_clinical_dict: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    n_steps: int = 50
) -> List[Dict[str, Any]]:
    """
    Computes genuine Integrated Gradients attributions for clinical features.
    
    Returns list of dictionaries containing:
    - feature: name ('age', 'sex', 'height', 'weight')
    - input_value: patient's actual measurement
    - attribution: mathematical contribution to abnormal-class logit
    - direction: 'toward_abnormal' or 'toward_normal'
    """
    explainer = PTBXLExplainer(model=model, device=device, n_steps=n_steps)
    explanation = explainer.explain(
        clinical_tensor=clinical_tensor,
        ecg_tensor=ecg_tensor,
        raw_clinical_dict=raw_clinical_dict
    )
    return explanation["clinical_features"]
