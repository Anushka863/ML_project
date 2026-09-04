import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.ptbxl_preprocessor import PTBXLPreprocessor
from app.models.clinical_encoder import ClinicalEncoder
from app.models.ecg_encoder import ECGEncoder
from app.models.multimodal_model import MultimodalFusion

class PTBXLDataset(Dataset):
    def __init__(self, meta_path, preprocessor):
        self.df = pd.read_csv(meta_path)
        self.preprocessor = preprocessor
        self.clinical_cols = preprocessor.clinical_cols
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Clinical
        clinical = torch.tensor(row[self.clinical_cols].values.astype('float32'))
        # Target
        target = torch.tensor(row['target'], dtype=torch.float32)
        # ECG
        filename_lr = row['filename_lr']
        ecg_signal = self.preprocessor.extract_waveform(filename_lr)
        ecg_sig_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
        
        return clinical, ecg_sig_tensor, target

class PTBXLClassifier(nn.Module):
    def __init__(self, clinical_in, ecg_channels, use_fusion=True):
        super().__init__()
        self.clinical_enc = ClinicalEncoder(input_dim=clinical_in, embedding_dim=16, hidden_dims=[32])
        self.ecg_enc = ECGEncoder(in_channels=ecg_channels, out_dim=128)
        self.fusion = MultimodalFusion(clinical_dim=16, image_dim=128, fused_dim=128)
        self.head = nn.Linear(128, 1) # Binary classification output
        
    def forward(self, clinical_x, ecg_x):
        c_emb = self.clinical_enc(clinical_x)
        e_emb = self.ecg_enc(ecg_x)
        f_emb = self.fusion(c_emb, e_emb, is_linked_patient=True)
        return self.head(f_emb).squeeze(-1)

def train_ptbxl():
    print("==================================================")
    print("PHASE 9 — TRAINING PIPELINE (PTB-XL MULTIMODAL)")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_dir = r"c:\Users\Aisha Fathima\OneDrive\Desktop\ML Project\ML_project"
    data_dir = os.path.join(base_dir, "data", "ptbxl")
    model_dir = os.path.join(base_dir, "models", "ptbxl_multimodal")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Init Preprocessor
    # We will assume data is extracted directly to data_dir
    extract_path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    
    # Just in case extraction structure is nested or not
    if not os.path.exists(extract_path):
         extract_path = data_dir
         
    preprocessor = PTBXLPreprocessor(data_dir=extract_path)
    
    # Check if preprocessed data exists. If not, generate splits
    train_meta = os.path.join(data_dir, "train_metadata.csv")
    if not os.path.exists(train_meta):
        preprocessor.process_data()
        
    # Dataset and Dataloader
    train_ds = PTBXLDataset(train_meta, preprocessor)
    val_ds = PTBXLDataset(os.path.join(data_dir, "val_metadata.csv"), preprocessor)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # Model
    model = PTBXLClassifier(clinical_in=4, ecg_channels=12).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 3 # 3 epochs for proof-of-concept
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (c_x, e_x, y) in enumerate(train_loader):
            c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(c_x, e_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_y = []
        with torch.no_grad():
            for c_x, e_x, y in val_loader:
                c_x, e_x, y = c_x.to(device), e_x.to(device), y.to(device)
                preds = model(c_x, e_x)
                val_loss += criterion(preds, y).item()
                probs = torch.sigmoid(preds)
                all_preds.extend((probs > 0.5).cpu().int().numpy())
                all_y.extend(y.cpu().int().numpy())
                
        acc = accuracy_score(all_y, all_preds)
        print(f"Epoch {epoch+1} Summary: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={acc:.4f}")
        
        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader),
            'val_acc': acc
        })
        
    print("Training finished.")
    
    # Save the standalone models
    torch.save(model.ecg_enc.state_dict(), os.path.join(model_dir, "ptbxl_ecg_encoder.pt"))
    torch.save(model.state_dict(), os.path.join(model_dir, "ptbxl_classifier.pt"))
    
    # Save training metrics
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(reports_dir, "training_metrics.csv"), index=False)
    
    with open(os.path.join(reports_dir, "training_summary.md"), "w") as f:
        f.write("# Training Summary\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Final Val Acc: {acc:.4f}\n")
        f.write("\nSaved weights successfully.")
        
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    train_ptbxl()
