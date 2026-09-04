import torch
import torch.nn as nn

class ECGEncoder(nn.Module):
    def __init__(self, in_channels=12, out_dim=128):
        """
        1D CNN for 12-lead ECG signals.
        Input shape: (Batch, 12, 1000) for 10-second 100Hz signals.
        """
        super(ECGEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1) # Global pooling
        )
        
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        x: (batch, 12, 1000)
        """
        if len(x.shape) != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected input shape (batch, 12, seq_len), got {x.shape}")
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1) # shape: (batch, 128)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
