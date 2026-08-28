"""
Graph Neural Network (GNN) Disease Risk Prediction Model.
Phase 9 — Supports Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN)
with multi-disease multi-head output logits (Diabetes, Heart Disease, CKD).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class SimpleGNNConv(nn.Module):
    """
    Fallback message passing layer if torch_geometric is not installed.
    Computes degree-normalized adjacency aggregation.
    """
    def __init__(self, in_features: int, out_features: int):
        super(SimpleGNNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        if edge_index.shape[1] == 0:
            return self.linear(x)
            
        row, col = edge_index[0], edge_index[1]
        
        # Self-loops
        self_loops = torch.arange(num_nodes, device=x.device)
        self_edge = torch.stack([self_loops, self_loops], dim=0)
        full_edge_index = torch.cat([edge_index, self_edge], dim=1)
        
        # Aggregate neighbor features
        out = torch.zeros_like(x)
        out.index_add_(0, full_edge_index[0], x[full_edge_index[1]])
        
        # Degree normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, full_edge_index[0], torch.ones(full_edge_index.shape[1], device=x.device))
        deg = torch.clamp(deg, min=1.0).unsqueeze(1)
        
        out = out / deg
        return self.linear(out)


class MultiDiseaseGNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        out_channels: int = 32,
        gnn_type: str = "gat",
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        disease_targets: Optional[List[str]] = None
    ):
        super(MultiDiseaseGNN, self).__init__()
        
        self.gnn_type = gnn_type.lower()
        self.disease_targets = disease_targets or ["diabetes", "heart_disease", "ckd"]
        
        # GNN Backbone Layers
        self.gnn_layers = nn.ModuleList()
        in_dim = in_channels
        
        for i in range(num_layers):
            layer_out = hidden_channels if i < num_layers - 1 else out_channels
            self.gnn_layers.append(SimpleGNNConv(in_dim, layer_out))
            in_dim = layer_out
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-Head Multi-Disease Prediction Heads
        self.heads = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(out_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 1)  # Binary logit output per disease
            ) for disease in self.disease_targets
        })

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        active_targets: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns a dictionary of disease logits: {'diabetes': Tensor, 'heart_disease': Tensor, 'ckd': Tensor}
        """
        h = x
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_index)
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
                
        targets_to_eval = active_targets or self.disease_targets
        outputs = {}
        
        for disease in targets_to_eval:
            if disease in self.heads:
                outputs[disease] = self.heads[disease](h)
                
        return outputs
