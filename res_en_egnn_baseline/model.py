import os
import sys
import torch
import torch.nn as nn

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.egnn.egnn_clean import EGNN
from atom3d.models.mlp import MLP

class ResEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers=4, device='cuda'):
        super().__init__()
        self.egnn = EGNN(in_node_nf=in_node_nf,  # Use the actual input dimension
                        hidden_nf=hidden_nf,
                        out_node_nf=hidden_nf,
                        n_layers=n_layers,
                        device=device)
        
        self.mlp = MLP(hidden_nf, [64], out_node_nf)
        self.to(device)

    def forward(self, h, x, edges, batch):
        # Apply EGNN to get node embeddings
        h, x = self.egnn(h, x, edges, edge_attr=None)
        
        # Select embeddings for central alpha carbons
        central_h = h[batch.ca_idx]
        
        # Predict residue class
        out = self.mlp(central_h)
        
        return out