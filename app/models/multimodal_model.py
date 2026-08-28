"""
Multimodal Patient Representation Fusion module.
Phase 7 — Fuses clinical feature embeddings and medical image embeddings for linked patient records.
"""
import torch
import torch.nn as nn
from typing import Optional


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        clinical_dim: int = 64,
        image_dim: int = 64,
        fused_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        super(MultimodalFusion, self).__init__()
        
        self.clinical_dim = clinical_dim
        self.image_dim = image_dim
        self.fused_dim = fused_dim
        
        concat_dim = clinical_dim + image_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU()
        )
        
        # Standalone clinical projector when image modality is unavailable
        self.clinical_projector = nn.Sequential(
            nn.Linear(clinical_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU()
        )

    def forward(
        self,
        clinical_emb: torch.Tensor,
        image_emb: Optional[torch.Tensor] = None,
        is_linked_patient: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        - If is_linked_patient is True and image_emb is provided: Concatenates and applies Fusion MLP.
        - Otherwise: Projects clinical_emb directly into fused_dim space without unlinked image pairing.
        """
        if is_linked_patient and image_emb is not None:
            if clinical_emb.shape[0] != image_emb.shape[0]:
                raise ValueError(f"Batch size mismatch: clinical {clinical_emb.shape[0]} vs image {image_emb.shape[0]}")
            cat = torch.cat([clinical_emb, image_emb], dim=1)
            return self.fusion_mlp(cat)
        else:
            # Fallback / Pure clinical representation for unlinked datasets
            return self.clinical_projector(clinical_emb)
