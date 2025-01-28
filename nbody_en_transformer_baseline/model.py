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
            charge: Node charges [batch_size, n_nodes]
            x: Node positions [batch_size, n_nodes, 3]
            vel: Node velocities [batch_size, n_nodes, 3]
        Returns:
            pred_pos: Predicted positions [batch_size, n_nodes, 3]
        """
        batch_size, n_nodes = charge.size()
        
        # Create batch-aware fully-connected edge index
        rows, cols = [], []
        for b in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        rows.append(b * n_nodes + i)
                        cols.append(b * n_nodes + j)
        edge_index = torch.tensor([rows, cols], device=self.device)
        
        # Reshape inputs for processing
        charge = charge.view(-1, 1)  # [batch_size * n_nodes, 1]
        x = x.view(-1, 3)  # [batch_size * n_nodes, 3]
        vel = vel.view(-1, 3)  # [batch_size * n_nodes, 3]
        
        # Project inputs to feature space
        h = self.charge_proj(charge) + self.vel_proj(vel)  # [batch_size * n_nodes, hidden_nf]
        
        # Apply transformer
        h, x_out = self.transformer(h, x, edge_index)
        
        # Predict position update
        pred_pos = self.pos_pred(h)
        
        # Reshape output back to batch form
        pred_pos = pred_pos.view(batch_size, n_nodes, 3)
        
        return pred_pos