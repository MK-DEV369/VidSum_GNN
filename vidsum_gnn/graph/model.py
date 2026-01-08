import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class VidSumGNN(nn.Module):
    """
    Graph Attention Network for Video Summarization
    Optimizations: pre-norm residuals, GELU activations, dropout, and safe init.
    """
    
    def __init__(
        self, 
        in_dim: int = 1536,
        hidden_dim: int = 1024, 
        num_heads: int = 8, 
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Safety check: hidden_dim must be divisible by num_heads
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # Core blocks
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # GATv2 layers (outputs hidden_dim when concat=True and head_dim=hidden_dim//num_heads)
        head_dim = hidden_dim // num_heads
        self.gat1 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads, dropout=dropout, concat=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads, dropout=dropout, concat=True
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Scoring head (raw scores, no sigmoid for better gradients)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Robust initialization
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # PyG convs have their own reset
        if hasattr(self.gat1, 'reset_parameters'):
            self.gat1.reset_parameters()
        if hasattr(self.gat2, 'reset_parameters'):
            self.gat2.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (optional, not used in current architecture)
        Returns:
            scores: Importance scores [num_nodes] (1D tensor matching labels)
        """
        # Input block (pre-norm + residual style)
        h = self.input_proj(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)
        
        # GAT Layer 1
        h1 = self.gat1(h, edge_index)
        h1 = self.norm2(h1)
        h1 = self.act(h1)
        h1 = self.drop(h1)
        h = h + h1
        
        # GAT Layer 2
        h2 = self.gat2(h, edge_index)
        h2 = self.norm3(h2)
        h2 = self.act(h2)
        h2 = self.drop(h2)
        h = h + h2
        
        # Scoring - CRITICAL: squeeze to match label shape [num_nodes]
        scores = self.scorer(h).squeeze(-1)
        # Return both scores and hidden representation for downstream fusion
        return scores, h
# High-impact ways to raise accuracy/F1 Class balance / threshold Try threshold sweep instead of fixed 0.6 to 
# maximize F1/accuracy on validation; log best threshold and reuse for test/inference. Slightly lower focal α (e.g., 0.65) 
# or γ (e.g., 1.5) to reduce over-focus on hard positives if recall is low. Top-k binarization ratio Tune the positive ratio 
# per graph (e.g., 10–20% grid) and choose the ratio yielding best val F1. Current 15% may not match label density. Learning rate 
# schedule Add a warmup or cosine decay; alternatively, step LR down on plateau with a larger patience for scheduler 
# (separate from early stop). Early stopping Decouple early stopping from fixed epochs: set patience to a smaller fixed number 
# (e.g., 8–10) but allow full 40 epochs; keep best checkpoint. Regularization and capacity Increase hidden dim (e.g., 640) or 
# heads (e.g., 6) if VRAM allows; add slight dropout on attention outputs (0.3→0.35) to reduce overfit. Edge construction Ensure 
# multimodal edges are enabled and informative; consider k-NN tuning (KNN_K 10→15) and edge_attr normalization if not already. 
# Sampler and class weights Revisit importance duplication: cap duplicates to avoid overfitting specific graphs; optionally add 
# per-node class weights (pos_weight in focal/BCE) based on per-batch positive ratio. Feature scaling Verify node feature normalization 
# (per-modality z-score) before fusion to prevent modality dominance; if absent, add a small LayerNorm after concatenation. 
# Validation rigor Use a held-out test split (currently 0%) to avoid optimism; shuffle seeds and average over 2–3 runs to confirm gains. 
# Concrete tweaks to try (fast loop) Sweep threshold in validation after each epoch (0.45–0.65 step 0.05) and report best F1/accuracy; 
# pick best for checkpoint saving. Sweep binarization ratio {0.10, 0.15, 0.20} for 5–8 epochs; keep best ratio. Reduce focal γ to 1.5; 
# if recall is already good, try γ=2.5 for precision. Increase hidden_dim to 640 and heads to 6 if VRAM is sufficient; otherwise keep 
# 512/4. Enable LR decay: start 1e-4, ReduceLROnPlateau with factor 0.5, patience 3.