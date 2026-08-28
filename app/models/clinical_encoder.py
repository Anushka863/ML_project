"""
Clinical Feature Encoder module.
Phase 5 — Maps preprocessed tabular clinical features into a configurable fixed-size embedding.
"""
import torch
import torch.nn as nn
from typing import List, Optional


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 64,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super(ClinicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        hidden_dims = hidden_dims or [128, 64]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Final embedding projection layer
        layers.append(nn.Linear(in_dim, embedding_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input shape: (batch_size, input_dim)
        Output shape: (batch_size, embedding_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        return self.encoder(x)
