import os
import sys
import torch
import torch.nn as nn

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.en_transformer.en_transformer import EnTransformer
from atom3d.models.mlp import MLP

class ResEnTransformer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_layers=4, n_heads=4, device='cuda'):
        super().__init__()
        
        # Main En Transformer network
        self.transformer = EnTransformer(
            input_nf=input_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        # Final MLP to predict residue class
        self.mlp = MLP(hidden_nf, [64], output_nf)
        
        self.to(device)

    def forward(self, h, x, edges, batch):
        """
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3]
        edges: Graph connectivity [2, n_edges]
        batch: Graph containing ca_idx for central residue position
        """
        # Apply En Transformer to get node embeddings for all atoms
        h, x = self.transformer(h, x, edges)
        
        # Select only the embeddings for central residues
        central_h = h[batch.ca_idx]
        
        # Predict residue class for central residues only 
        out = self.mlp(central_h)
        
        return out