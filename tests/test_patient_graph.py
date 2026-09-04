import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.patient_graph import PatientGraphBuilder, HAS_PYG
try:
    from torch_geometric.data import Data
except ImportError:
    pass

def test_knn_graph_construction():
    builder = PatientGraphBuilder(k=3)
    N = 10
    dim = 16
    embeddings = torch.randn(N, dim)
    labels = torch.randint(0, 2, (N,))
    
    data = builder.build_knn_graph(embeddings, labels)
    
    if HAS_PYG:
        assert isinstance(data, Data)
        assert data.x.shape == (N, dim)
        assert data.y.shape == (N,)
        # Each of N nodes has exactly 3 neighbors (since k=3)
        assert data.edge_index.shape[1] == N * 3
    else:
        assert isinstance(data, dict)
        assert data['x'].shape == (N, dim)
        assert data['y'].shape == (N,)
        assert data['edge_index'].shape[1] == N * 3
        
    print("test_knn_graph_construction passed!")

def test_graph_statistics():
    builder = PatientGraphBuilder(k=2)
    embeddings = torch.randn(5, 8)
    labels = torch.tensor([0, 1, 0, 1, 0])
    
    data = builder.build_knn_graph(embeddings, labels)
    num_nodes = builder.get_num_nodes(data)
    num_edges = builder.get_num_edges(data)
    
    assert num_nodes == 5
    assert num_edges == 5 * 2
    print("test_graph_statistics passed!")

if __name__ == "__main__":
    test_knn_graph_construction()
    test_graph_statistics()
