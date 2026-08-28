"""
Image Explainability (XAI) module.
Phase 13 — Implements Grad-CAM heatmap visualization for CNN image backbones.
Labels highlighted areas as 'model-attended regions contributing to prediction'.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional


class ImageGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmap for an image tensor (1, 3, H, W).
        Returns normalized heatmap array and clinical disclaimer label.
        """
        self.model.eval()
        image_tensor.requires_grad = True
        
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        self.model.zero_grad()
        score.backward()
        
        # Calculate pooled gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize between 0 and 1
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_norm = np.zeros_like(cam_np)

        return {
            "heatmap": cam_norm,
            "target_class": int(class_idx),
            "label_disclaimer": "Highlighted areas represent model-attended regions contributing to prediction. They do not constitute confirmed disease pathology."
        }
