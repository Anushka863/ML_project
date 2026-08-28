"""
Graph Explainability (XAI) module.
Phase 14 — Explains patient similarity graph edges and node neighbor attributions.
Clearly labels explanations as computational model representations, not social or causal relationships.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List


class GraphXAI:
    def __init__(self, gnn_model: nn.Module):
        self.gnn_model = gnn_model

    def explain_patient_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_disease: str = "diabetes"
    ) -> Dict[str, Any]:
        """
        Identify top computational neighbor nodes contributing to target patient node prediction.
        """
        self.gnn_model.eval()
        
        # Find edges connected to target node_idx
        row, col = edge_index[0], edge_index[1]
        connected_mask = (row == node_idx) | (col == node_idx)
        connected_edges = edge_index[:, connected_mask]
        
        neighbors = []
        if connected_edges.shape[1] > 0:
            neighbor_nodes = set(connected_edges.flatten().tolist()) - {node_idx}
            for n_idx in neighbor_nodes:
                # Cosine similarity between target node and neighbor
                sim = float(torch.nn.functional.cosine_similarity(
                    x[node_idx].unsqueeze(0), x[n_idx].unsqueeze(0)
                ).item())
                neighbors.append({
                    "neighbor_node_id": int(n_idx),
                    "feature_similarity_score": round(sim, 4),
                    "explanation": f"Patient #{n_idx} shares similar clinical profile embedding."
                })
                
        neighbors.sort(key=lambda item: item["feature_similarity_score"], reverse=True)

        return {
            "target_patient_id": int(node_idx),
            "target_disease": target_disease,
            "top_similar_neighbors": neighbors[:5],
            "disclaimer": "Graph edges represent computational similarity between patient representations. They do not represent family, social, causal or direct medical relationships."
        }
