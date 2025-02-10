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
    E(n) Transformer model for N-body simulation prediction that maintains E(n) equivariance
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
        
        # Input projections for invariant features
        self.charge_proj = nn.Linear(1, hidden_nf)
        
        # Process velocities in an equivariant way
        self.vel_norm = nn.Sequential(
            nn.Linear(1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Main transformer
        self.transformer = EnTransformer(
            input_nf=hidden_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        # Final position prediction (maintains equivariance)
        self.pos_scale = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
        self.to(device)
        
    def forward(self, charge, x, vel):
        """
        Args:
            charge: Node charges [batch_size, n_nodes] (invariant)
            x: Node positions [batch_size, n_nodes, 3] (equivariant)
            vel: Node velocities [batch_size, n_nodes, 3] (equivariant)
        Returns:
            pred_pos: Predicted positions [batch_size, n_nodes, 3] (equivariant)
        """
        # Add shape check
        if len(charge.shape) != 2:
            raise ValueError(f"Expected charge to have 2 dimensions [batch_size, n_nodes], got shape {charge.shape}")
        
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
        
        # Process invariant features
        charge = charge.view(-1, 1)  # [batch_size * n_nodes, 1]
        
        # Process equivariant features
        x = x.view(-1, 3)  # [batch_size * n_nodes, 3]
        vel = vel.view(-1, 3)  # [batch_size * n_nodes, 3]
        
        # Compute velocity norm (invariant)
        vel_norm = torch.norm(vel, dim=-1, keepdim=True)  # [batch_size * n_nodes, 1]
        
        # Create invariant node features
        h = self.charge_proj(charge) + self.vel_norm(vel_norm)  # [batch_size * n_nodes, hidden_nf]
        
        # Apply transformer (maintains equivariance)
        h, x_out = self.transformer(h, x, edge_index)
        
        # Scale the coordinate differences in an equivariant way
        scale = self.pos_scale(h).view(-1, 1)  # [batch_size * n_nodes, 1]
        pred_pos = x_out + scale * vel  # Equivariant update
        
        # Reshape output back to batch form
        pred_pos = pred_pos.view(batch_size, n_nodes, 3)
        
        return pred_pos