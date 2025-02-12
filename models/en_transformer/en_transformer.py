import torch
import torch.nn as nn
import torch.nn.functional as F

class EnAttention(nn.Module):
    """
    E(n) Equivariant Attention Layer that combines coordinate and feature attention
    while preserving E(n) equivariance.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_heads = n_heads
        self.dim_head = dim_head
        
        # Feature attention components
        self.to_qkv = nn.Linear(input_nf, 3 * n_heads * dim_head)
        self.to_out = nn.Linear(n_heads * dim_head, output_nf)
        
        # Coordinate attention components
        self.coors_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1),
        )
        
        # Edge network
        edge_input_dim = hidden_nf + 1  # +1 for radial distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )

    def forward(self, h, x, edge_index):
        """
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3]
        edge_index: Graph connectivity [2, n_edges]
        """
        row, col = edge_index
        
        # Get relative positional encodings
        rel_pos = x[row] - x[col]  # [n_edges, 3]
        rel_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)  # [n_edges, 1]
        
        # Process edges
        edge_attr = torch.cat([h[row], h[col], rel_dist], dim=-1)  # [n_edges, hidden_nf + 1]
        edge_features = self.edge_mlp(edge_attr)  # [n_edges, hidden_nf]
        
        # Feature attention
        qkv = self.to_qkv(h).chunk(3, dim=-1)  # [n_nodes, n_heads * dim_head] each
        q, k, v = map(lambda t: t.view(-1, self.n_heads, self.dim_head), qkv)
        
        # Compute attention scores
        dots = torch.einsum('bhd,bjd->bhj', q, k) * (self.dim_head ** -0.5)
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        feats = torch.einsum('bhj,bjd->bhd', attn, v)
        feats = feats.reshape(-1, self.n_heads * self.dim_head)
        feats = self.to_out(feats)  # [n_nodes, output_nf]
        
        # Coordinate attention
        edge_weights = self.coors_mlp(edge_features)  # [n_edges, 1]

        # Update coordinates while preserving E(n) equivariance
        coor_diff = edge_weights * rel_pos  # [n_edges, 3]
        coor_out = torch.zeros_like(x)  # [n_nodes, 3]
        coor_out.index_add_(0, row, coor_diff)
        
        return feats, coor_out

class EnTransformerLayer(nn.Module):
    """
    Complete E(n)-Transformer layer with attention and feedforward.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.attention = EnAttention(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(output_nf, hidden_nf * 2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, output_nf)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_nf)
        self.norm2 = nn.LayerNorm(output_nf)

    def forward(self, h, x, edge_index):
        # Attention
        h_norm = self.norm1(h)
        h_attn, x_update = self.attention(h_norm, x, edge_index)
        h = h + h_attn  # Residual connection
        x = x + x_update  # Update coordinates
        
        # Feedforward
        h = h + self.ff(self.norm2(h))  # Residual connection
        
        return h, x

class EnTransformer(nn.Module):
    """
    Full E(n)-Transformer model with multiple layers.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_layers=4, n_heads=4, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList([
            EnTransformerLayer(
                input_nf if i == 0 else hidden_nf,
                hidden_nf if i < n_layers - 1 else output_nf,
                hidden_nf,
                n_heads,
                dim_head
            )
            for i in range(n_layers)
        ])

    def forward(self, h, x, edge_index):
        """
        h: Input node features [n_nodes, input_nf]
        x: Input coordinates [n_nodes, 3] 
        edge_index: Graph connectivity [2, n_edges]
        """
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return h, x