import os
import torch
import numpy as np
import scipy.sparse as sp

from typing import Tuple, Optional

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class PatientGraphBuilder:
    def __init__(self, k: int = 5, k_neighbors: Optional[int] = None, metric: str = "cosine"):
        self.k = k_neighbors if k_neighbors is not None else k
        self.k_neighbors = self.k
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
            norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.mm(norm_emb, norm_emb.t())
        elif self.metric == "euclidean":
            dist = torch.cdist(embeddings, embeddings, p=2)
            sim_matrix = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unsupported similarity metric '{self.metric}'. Choose 'cosine' or 'euclidean'.")

        sim_matrix.fill_diagonal_(-float("inf"))
        
        topk_sim, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        
        row_indices = torch.arange(num_patients, device=embeddings.device).unsqueeze(1).repeat(1, k).view(-1)
        col_indices = topk_indices.view(-1)
        weights = topk_sim.view(-1, 1)
        
        edge_index = torch.stack([row_indices, col_indices], dim=0)
        
        return edge_index, weights, embeddings

    def build_knn_graph(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Builds a K-Nearest Neighbor graph based on cosine similarity of the embeddings.
        embeddings: (N, D) Tensor
        labels: (N,) Tensor
        Returns PyTorch Geometric Data object (or dict if PyG not installed).
        """
        N = embeddings.shape[0]
        emb_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        
        sim_matrix = torch.matmul(emb_norm, emb_norm.t())
        topk_vals, topk_inds = torch.topk(sim_matrix, k=min(self.k + 1, N), dim=1)
        
        edge_index_sources = []
        edge_index_targets = []
        
        for i in range(N):
            for j in range(1, min(self.k + 1, N)):  # Skip 0 (self)
                neighbor = topk_inds[i][j].item()
                edge_index_sources.append(i)
                edge_index_targets.append(neighbor)
                
        edge_index = torch.tensor([edge_index_sources, edge_index_targets], dtype=torch.long)
        
        if HAS_PYG:
            data = Data(x=embeddings, edge_index=edge_index, y=labels)
        else:
            data = {
                'x': embeddings,
                'edge_index': edge_index,
                'y': labels
            }
            
        return data

    def get_num_nodes(self, data):
        if HAS_PYG and isinstance(data, Data):
            return data.num_nodes
        return data['x'].shape[0]

    def get_num_edges(self, data):
        if HAS_PYG and isinstance(data, Data):
            return data.num_edges
        return data['edge_index'].shape[1]

    def get_avg_degree(self, data):
        num_nodes = self.get_num_nodes(data)
        num_edges = self.get_num_edges(data)
        return num_edges / num_nodes if num_nodes > 0 else 0

    def generate_statistics(self, data, save_path="reports/patient_graph_stats.md"):
        num_nodes = self.get_num_nodes(data)
        num_edges = self.get_num_edges(data)
        avg_degree = self.get_avg_degree(data)
        
        edge_index = data.edge_index if HAS_PYG and isinstance(data, Data) else data['edge_index']
        
        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        ones = np.ones(len(row))
        
        adj_matrix = sp.coo_matrix((ones, (row, col)), shape=(num_nodes, num_nodes))
        n_components, component_labels = sp.csgraph.connected_components(adj_matrix, directed=False)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("# Patient Similarity Graph Statistics\n\n")
            f.write(f"- **Number of nodes**: {num_nodes}\n")
            f.write(f"- **Number of edges**: {num_edges}\n")
            f.write(f"- **Average degree**: {avg_degree:.2f}\n")
            f.write(f"- **Connected components**: {n_components}\n")
            f.write(f"- **k (nearest neighbors)**: {self.k}\n")
            f.write("\nGraph constructed using fused multimodal embeddings (Clinical + ECG) via Cosine Similarity.\n")
        
        print(f"Graph statistics generated at {save_path}")

def save_graph(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(data, filepath)
    print(f"Graph saved to {filepath}")
