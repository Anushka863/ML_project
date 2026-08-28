"""
Medical Image Feature Encoder module.
Phase 6 — Uses pretrained CNN backbones (ResNet, EfficientNet) to extract fixed-size image embeddings.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 64,
        freeze_backbone: bool = False
    ):
        super(ImageEncoder, self).__init__()
        
        self.backbone_name = backbone_name.lower()
        self.embedding_dim = embedding_dim
        
        weights = "DEFAULT" if pretrained else None
        
        if self.backbone_name == "resnet18":
            resnet = models.resnet18(weights=weights)
            in_features = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        elif self.backbone_name == "resnet50":
            resnet = models.resnet50(weights=weights)
            in_features = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        elif self.backbone_name == "efficientnet_b0":
            effnet = models.efficientnet_b0(weights=weights)
            in_features = effnet.classifier[1].in_features
            effnet.classifier = nn.Identity()
            self.backbone = effnet
        else:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from 'resnet18', 'resnet50', 'efficientnet_b0'")
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Embedding projection head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape: (batch_size, 3, H, W)
        Output shape: (batch_size, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return embeddings
