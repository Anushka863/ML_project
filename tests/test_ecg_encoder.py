import os
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.ecg_encoder import ECGEncoder

def test_ecg_encoder_forward():
    model = ECGEncoder(in_channels=12, out_dim=128)
    # Create fake batch of 4 patients, 12 leads, 1000 sequence length (10 seconds @ 100Hz)
    dummy_input = torch.randn(4, 12, 1000)
    
    output = model(dummy_input)
    assert output.shape == (4, 128), f"Expected shape (4, 128), got {output.shape}"
    print("test_ecg_encoder_forward passed!")

def test_ecg_encoder_save_load():
    model = ECGEncoder()
    path = "dummy_ecg_encoder.pt"
    model.save(path)
    
    assert os.path.exists(path), "Model was not saved."
    
    model_new = ECGEncoder()
    model_new.load(path)
    
    os.remove(path)
    print("test_ecg_encoder_save_load passed!")

if __name__ == "__main__":
    test_ecg_encoder_forward()
    test_ecg_encoder_save_load()
