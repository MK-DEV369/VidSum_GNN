import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class GraphBuilder:
    def __init__(self, k_sim: int = 5, sim_threshold: float = 0.65, max_edges: int = 20):
        self.k_sim = k_sim
        self.sim_threshold = sim_threshold
        self.max_edges = max_edges

    def build_graph(self, shots: List[Dict], features: torch.Tensor) -> Data:
        """
        Build a graph from shots and features.
        shots: List of dicts with 'start_sec', 'end_sec', etc.
        features: Tensor of shape (num_shots, feature_dim)
        """
        num_nodes = len(shots)
        edge_index = []
        edge_attr = []

        # 1. Temporal Edges (i <-> i+1, i <-> i+2)
        for i in range(num_nodes):
            for offset in [1, 2]:
                j = i + offset
                if j < num_nodes:
                    # Bidirectional
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    dist = shots[j]['start_sec'] - shots[i]['start_sec']
                    # Attr: [is_temporal, distance, sim, audio_corr]
                    # We'll normalize distance roughly
                    attr = [1.0, dist / 60.0, 0.0, 0.0] 
                    edge_attr.append(attr)
                    edge_attr.append(attr)

        # 2. Semantic Similarity Edges
        # Compute pairwise cosine similarity
        # features is (N, D)
        # normalize features first
        feats_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(feats_norm, feats_norm.t()) # (N, N)
        
        # Remove self-loops for top-k
        sim_matrix.fill_diagonal_(-1.0)
        
        values, indices = torch.topk(sim_matrix, k=min(self.k_sim, num_nodes-1), dim=1)
        
        for i in range(num_nodes):
            for k in range(indices.shape[1]):
                j = indices[i, k].item()
                sim = values[i, k].item()
                
                if sim > self.sim_threshold:
                    edge_index.append([i, j])
                    # Attr: [is_temporal, distance, sim, audio_corr]
                    attr = [0.0, 0.0, sim, 0.0]
                    edge_attr.append(attr)

        # 3. Audio Edges (Placeholder logic)
        # If we had audio features separate, we'd do similar correlation check.
        # For now, we skip or assume fused features cover it.

        if not edge_index:
            # Fallback: just temporal
            edge_index = [[0, 0]]  # Self loop to avoid crash if single node
            edge_attr = [[1.0, 0.0, 0.0, 0.0]]
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(x=features, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
        
        return data
