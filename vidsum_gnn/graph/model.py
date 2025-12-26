import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class VidSumGNN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int = 1024, 
        num_heads: int = 8, 
        dropout: float = 0.2,
        edge_dim: int = 4
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GATv2 Layer 1
        # heads * out_channels = hidden_dim usually, or we project.
        # Spec: 8 heads, 128 per head -> 1024 total.
        head_dim = hidden_dim // num_heads
        self.gat1 = GATv2Conv(
            hidden_dim, 
            head_dim, 
            heads=num_heads, 
            dropout=dropout, 
            edge_dim=edge_dim,
            concat=True
        )
        
        # GATv2 Layer 2
        self.gat2 = GATv2Conv(
            hidden_dim, # input is heads*head_dim = 1024
            head_dim, 
            heads=num_heads, 
            dropout=dropout, 
            edge_dim=edge_dim,
            concat=True
        )
        
        # Readout / Scoring
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_dim)
        x = self.input_proj(x)
        x = F.elu(x)
        
        # Layer 1
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # Layer 2
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # Scoring
        scores = self.mlp(x) # (N, 1)
        
        return scores
