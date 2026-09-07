"""
==============================================================================
PTB-XL Patient Similarity Graph Construction (Phase 5)
==============================================================================
Constructs patient similarity graphs from genuine multimodal patient representations
and exports graph statistics to reports.
==============================================================================
"""

import os
import sys

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from app.graph.patient_graph import PatientGraphBuilder, save_graph
from ml_pipeline.train_ptbxl_multimodal import PTBXLDataset, PTBXLMultimodalGNN


def run_build_graph(num_patients: int = 500):
    print("=" * 70)
    print("🕸️ BUILDING PTB-XL PATIENT SIMILARITY GRAPH")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = PROJECT_ROOT / "data" / "ptbxl"
    graph_dir = PROJECT_ROOT / "data" / "processed" / "graph"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models" / "ptbxl_multimodal"
    
    graph_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    extract_path = data_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    if not extract_path.exists():
        extract_path = data_dir
        
    preprocessor = PTBXLPreprocessor(data_dir=str(extract_path))
    val_csv = data_dir / "val_metadata.csv"
    
    if not val_csv.exists():
        preprocessor.process_data()
        
    ds = PTBXLDataset(str(val_csv), preprocessor, max_samples=num_patients)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # Load trained model to get genuine fused representations
    model = PTBXLMultimodalGNN(
        clinical_in=4,
        ecg_channels=12,
        clinical_emb_dim=64,
        ecg_emb_dim=128,
        fused_dim=128,
        gnn_hidden=64,
        gnn_out=32,
        k_neighbors=5
    ).to(device)
    
    checkpoint_path = models_dir / "best_model.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded weights from {checkpoint_path}")
        
    model.eval()
    
    all_fused = []
    all_targets = []
    
    with torch.no_grad():
        for c_x, e_x, y in loader:
            c_x, e_x = c_x.to(device), e_x.to(device)
            c_emb = model.clinical_enc(c_x)
            e_emb = model.ecg_enc(e_x)
            f_emb = model.fusion(c_emb, e_emb, is_linked_patient=True)
            
            all_fused.append(f_emb.cpu())
            all_targets.append(y)
            
    fused_embeddings = torch.cat(all_fused, dim=0)
    labels = torch.cat(all_targets, dim=0).long()
    
    print(f"Constructing k-NN similarity graph for {len(fused_embeddings)} patient recordings...")
    builder = PatientGraphBuilder(k=5)
    graph_data = builder.build_knn_graph(fused_embeddings, labels)
    
    save_path = graph_dir / "ptbxl_graph.pt"
    save_graph(graph_data, str(save_path))
    
    stats_path = reports_dir / "patient_graph_stats.md"
    builder.generate_statistics(graph_data, save_path=str(stats_path))
    print("Graph build complete!")


if __name__ == "__main__":
    run_build_graph()
