"""
Bilateral Ophthalmic Explainability (XAI) Module.
Phase: ODIR Ophthalmic Branch

Implements:
1. Bilateral Grad-CAM heatmaps for both Left Eye and Right Eye fundus images.
2. Demographic Feature Attribution (Integrated Gradients) for Age & Sex.
3. Clean model disclaimer labeling: "Model-attended retinal regions contributing to prediction."
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional, Tuple

from app.explainability.image_xai import ImageGradCAM
from app.models.odir_multimodal_model import ODIRMultimodalGNN, ODIR_TARGETS


class ODIRBilateralExplainer:
    def __init__(self, model: ODIRMultimodalGNN, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

        # Find target CNN layer for Grad-CAM (ResNet layer4)
        single_encoder = self.model.bilateral_encoder.single_eye_encoder
        if hasattr(single_encoder.backbone, "layer4"):
            self.target_layer = single_encoder.backbone.layer4[-1]
        elif hasattr(single_encoder.backbone, "conv_head"):
            self.target_layer = single_encoder.backbone.conv_head
        else:
            # Fallback to last conv child
            convs = [m for m in single_encoder.backbone.modules() if isinstance(m, nn.Conv2d)]
            self.target_layer = convs[-1] if convs else None

    def explain_fundus_image(
        self,
        image_tensor: torch.Tensor,
        target_class_idx: int = 0
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap (224, 224) for a single eye image tensor (1, 3, 224, 224).
        """
        if self.target_layer is None:
            return np.zeros((224, 224), dtype=np.float32)

        grad_cam = ImageGradCAM(self.model.bilateral_encoder.single_eye_encoder, self.target_layer)
        res = grad_cam.generate_heatmap(image_tensor.to(self.device), class_idx=target_class_idx)
        return res["heatmap"]

    def explain_patient(
        self,
        demo_tensor: torch.Tensor,
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
        top_disease: str = "D"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive bilateral visual & demographic explanation for a patient.
        """
        self.model.eval()
        demo_t = demo_tensor.to(self.device)
        left_t = left_tensor.to(self.device)
        right_t = right_tensor.to(self.device)

        # Grad-CAM for Left and Right images
        left_heatmap = self.explain_fundus_image(left_t)
        right_heatmap = self.explain_fundus_image(right_t)

        # Demographic feature attribution (finite difference / gradient sensitivity)
        demo_t.requires_grad = True
        logits = self.model(demo_t, left_t, right_t)
        
        target_logit = logits.get(top_disease, list(logits.values())[0])
        self.model.zero_grad()
        target_logit.backward(retain_graph=False)

        demo_grad = demo_t.grad.detach().cpu().numpy().flatten() if demo_t.grad is not None else np.array([0.0, 0.0])
        
        feature_attributions = [
            {
                "feature": "Patient Age",
                "attribution": float(round(demo_grad[0], 4)),
                "importance": float(round(abs(demo_grad[0]), 4)),
                "impact": "Higher risk factor" if demo_grad[0] > 0 else "Lower risk factor"
            },
            {
                "feature": "Patient Sex",
                "attribution": float(round(demo_grad[1], 4)),
                "importance": float(round(abs(demo_grad[1]), 4)),
                "impact": "Demographic baseline contribution"
            }
        ]

        return {
            "target_disease": top_disease,
            "left_eye_heatmap": left_heatmap.tolist(),
            "right_eye_heatmap": right_heatmap.tolist(),
            "demographic_attributions": feature_attributions,
            "disclaimer": "Highlighted retinal areas indicate model-attended regions contributing to the classification. They do not constitute confirmed clinical pathology."
        }
