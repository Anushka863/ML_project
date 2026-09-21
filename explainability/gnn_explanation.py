"""
==============================================================================
Graph Explanation Module for PTB-XL Inference
==============================================================================
Reports genuine graph structure during inference. Single-patient intake
creates an isolated N=1 node without graph edges; message passing between
patients does NOT occur. This module reports the honest architecture state.
==============================================================================
"""

from typing import Dict, Any, List, Optional
import torch


def explain_inference_graph(
    num_nodes: int = 1,
    num_edges: int = 0,
    edge_index: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Explains the inference graph configuration without fabricating neighbors or edges.
    """
    if edge_index is not None and edge_index.ndim == 2:
        edges_count = edge_index.shape[1]
    else:
        edges_count = num_edges

    has_message_passing = bool(edges_count > 0 and num_nodes > 1)

    if not has_message_passing:
        detail = (
            f"Graph contains {num_nodes} node(s) and {edges_count} edges. "
            "No cross-patient message passing occurred during inference. "
            "The GNN layer acts as a direct feature projection without neighbor aggregation."
        )
    else:
        detail = (
            f"Graph contains {num_nodes} nodes and {edges_count} edges with active k-NN message passing."
        )

    return {
        "nodes": num_nodes,
        "edges": edges_count,
        "target_node": 0,
        "neighbors": [],
        "message_passing": has_message_passing,
        "explanation": detail
    }
