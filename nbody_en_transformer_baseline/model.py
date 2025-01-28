import torch
import torch.nn as nn
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.en_transformer.en_transformer import EnTransformer

class NBodyTransformer(nn.Module):
    """
    E(n) Transformer model for N-body simulation prediction
    """
    def __init__(
        self,
        hidden_nf=64,
        n_layers=4,
        n_heads=4,
        device='cuda'
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        
        # Input projections
        self.charge_proj = nn.Linear(1, hidden_nf)
        self.vel_proj = nn.Linear(3, hidden_nf)
        
        # Main transformer
        self.transformer = EnTransformer(
            input_nf=hidden_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        # Final position prediction
        self.pos_pred = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3)
        )
        
        self.to(device)
        
    def forward(self, charge, x, vel):
        """
        Args:
            charge: Node charges [n_nodes, 1]
            x: Node positions [n_nodes, 3]
            vel: Node velocities [n_nodes, 3]
        Returns:
            pred_pos: Predicted positions [n_nodes, 3]
        """
        batch_size = charge.size(0)
        
        # Create fully-connected edge index
        n_nodes = charge.size(0)
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edge_index = torch.tensor([rows, cols], device=self.device)
        
        # Project inputs to feature space
        h = self.charge_proj(charge.unsqueeze(-1)) + self.vel_proj(vel)
        
        # Apply transformer
        h, x_out = self.transformer(h, x, edge_index)
        
        # Predict position update
        pred_pos = self.pos_pred(h)
        
        return pred_pos