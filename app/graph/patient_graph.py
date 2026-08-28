"""
Patient Similarity Graph Builder module.
Phase 8 — Constructs a k-Nearest Neighbors (k-NN) graph based on cosine similarity
of patient feature embeddings.

DOCUMENTATION NOTICE:
Graph edges represent computational similarity between patient representations.
They do NOT represent family, social, causal, or direct medical relationships.
"""
import torch
import numpy as np
from typing import Tuple, Optional


class PatientGraphBuilder:
    def __init__(self, k_neighbors: int = 5, metric: str = "cosine"):
        self.k_neighbors = k_neighbors
        self.metric = metric.lower()

    def build_graph(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct a k-NN similarity graph from patient embeddings.
        
        Args:
            embeddings: Tensor of shape (num_patients, feature_dim)
            
        Returns:
            edge_index: PyTorch Geometric format edge tensor of shape (2, num_edges)
            edge_attr: Edge similarity weights tensor of shape (num_edges, 1)
            x: Node features tensor of shape (num_patients, feature_dim)
        """
        num_patients = embeddings.shape[0]
        if num_patients <= 1:
            # Single patient edge-less graph (e.g. for single-patient inference)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=embeddings.device)
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=embeddings.device)
            return edge_index, edge_attr, embeddings
            
        k = min(self.k_neighbors, num_patients - 1)
        
        if self.metric == "cosine":
            # Normalize embeddings to unit length for cosine similarity computation
            norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.mm(norm_emb, norm_emb.t())  # Shape: (N, N)
        elif self.metric == "euclidean":
            dist = torch.cdist(embeddings, embeddings, p=2)
            sim_matrix = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unsupported similarity metric '{self.metric}'. Choose 'cosine' or 'euclidean'.")

        # Zero out self-loops in KNN computation
        sim_matrix.fill_diagonal_(-float("inf"))
        
        # Get top-k nearest neighbors for each node
        topk_sim, topk_indices = torch.topk(sim_matrix, k=k, dim=1)  # Shapes: (N, k)
        
        row_indices = torch.arange(num_patients, device=embeddings.device).unsqueeze(1).repeat(1, k).view(-1)
        col_indices = topk_indices.view(-1)
        weights = topk_sim.view(-1, 1)
        
        edge_index = torch.stack([row_indices, col_indices], dim=0)  # Shape: (2, N*k)
        
        return edge_index, weights, embeddings
