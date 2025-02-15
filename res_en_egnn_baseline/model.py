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
        self.egnn = EGNN(in_node_nf=in_node_nf, 
                        hidden_nf=hidden_nf,
                        out_node_nf=hidden_nf,
                        n_layers=n_layers,
                        device=device)
        
        # Final MLP to predict residue class
        self.mlp = MLP(hidden_nf, [hidden_nf], out_node_nf)
        
        self.to(device)

    def forward(self, h, x, edges, central_indices):
        """
        h: Node features [n_nodes, in_node_nf]
        x: Node coordinates [n_nodes, 3] 
        edges: Graph connectivity [2, n_edges]
        central_indices: Indices of central residues [n_central_residues]
        """
        # Apply EGNN to get node embeddings for all atoms
        h, x = self.egnn(h, x, edges, edge_attr=None)
        
        # Select only the embeddings for central residues
        central_h = h[central_indices]
        
        # Predict residue class for central residues only
        out = self.mlp(central_h)
        
        return out