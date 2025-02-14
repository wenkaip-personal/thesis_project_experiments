import torch
import torch.nn as nn
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.en_transformer.en_transformer import EnAttention, EnTransformerLayer, EnTransformer

class EnAttention_vel(EnAttention):
    """
    E(n) Equivariant Attention Layer that incorporates velocity information
    while preserving E(n) equivariance
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        # Additional velocity MLP
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )

    def forward(self, h, x, vel, edge_index):
        """
        h: Node features [n_nodes, input_nf] (invariant)
        x: Node coordinates [n_nodes, 3] (equivariant)
        vel: Node velocities [n_nodes, 3] (equivariant)
        edge_index: Graph connectivity [2, n_edges]
        """
        row, col = edge_index
        
        # Process edges as in base class
        rel_pos = x[row] - x[col]
        rel_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        
        # Process edges
        edge_attr = torch.cat([h[row], h[col], rel_dist], dim=-1)
        edge_features = self.edge_mlp(edge_attr)
        
        # Feature attention (invariant)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(-1, self.n_heads, self.dim_head), qkv)
        
        dots = torch.einsum('bhd,bjd->bhj', q, k) * (self.dim_head ** -0.5)
        attn = torch.softmax(dots, dim=-1)
        
        feats = torch.einsum('bhj,bjd->bhd', attn, v)
        feats = feats.reshape(-1, self.n_heads * self.dim_head)
        feats = self.to_out(feats)
        
        # Coordinate attention (equivariant)
        edge_weights = self.coors_mlp(edge_features)
        coord_weights = torch.softmax(edge_weights, dim=0)
        
        # Update coordinates (equivariant)
        coor_diff = coord_weights * rel_pos
        coor_out = torch.zeros_like(x)
        coor_out.index_add_(0, row, coor_diff)
        coor_out += self.coord_mlp_vel(h) * vel
        
        return feats, coor_out

class EnTransformerLayer_vel(nn.Module):
    """
    Complete E(n)-Transformer layer with velocity handling
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.attention = EnAttention_vel(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        self.ff = nn.Sequential(
            nn.Linear(output_nf, hidden_nf * 2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, output_nf)
        )
        
        self.norm1 = nn.LayerNorm(input_nf)
        self.norm2 = nn.LayerNorm(output_nf)

    def forward(self, h, x, vel, edge_index):
        # Attention with velocity
        h_norm = self.norm1(h)
        h_attn, x_update = self.attention(h_norm, x, vel, edge_index)
        h = h + h_attn  # Residual for features (invariant)
        x = x + x_update  # Update coordinates (equivariant)
        
        # Feedforward on features (invariant)
        h = h + self.ff(self.norm2(h))
        
        return h, x

class NBodyTransformer_vel(nn.Module):
    """
    E(n) Transformer model for N-body simulation that handles velocities directly
    """
    def __init__(self, hidden_nf=64, n_layers=4, n_heads=4, dim_head=64, device='cuda'):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        
        # Process velocities in an equivariant way
        self.vel_norm = nn.Sequential(
            nn.Linear(1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            EnTransformerLayer_vel(
                hidden_nf if i > 0 else hidden_nf,
                hidden_nf,
                hidden_nf,
                n_heads,
                dim_head
            )
            for i in range(n_layers)
        ])

        self.to(device)

    def forward(self, charge, x, vel):
        """
        charge: Node charges [batch_size, n_nodes] (invariant)
        x: Node positions [batch_size, n_nodes, 3] (equivariant) 
        vel: Node velocities [batch_size, n_nodes, 3] (equivariant)
        """
        batch_size, n_nodes = charge.size()
        
        # Create edge connectivity
        rows, cols = [], []
        for b in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        rows.append(b * n_nodes + i)
                        cols.append(b * n_nodes + j)
        edge_index = torch.tensor([rows, cols], device=self.device)
        
        # Reshape inputs
        x = x.view(-1, 3)
        vel = vel.view(-1, 3)
        
        # Create initial node features (invariant)
        vel_norm = torch.norm(vel, dim=-1, keepdim=True)
        h = self.vel_norm(vel_norm)
        
        # Apply transformer layers
        for layer in self.layers:
            h, x = layer(h, x, vel, edge_index)
            
        # Reshape outputs back to batch form
        x = x.view(batch_size, n_nodes, 3)
        
        return x