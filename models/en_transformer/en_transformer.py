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
        edge_input_dim = input_nf * 2 + 1  # +1 for radial distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )

    def forward(self, h, x, edge_index, mask=None):
        """
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3]
        edge_index: Graph connectivity [2, n_edges]
        mask: Attention mask [n_nodes, n_nodes] or None
        """
        n_nodes = h.size(0)
        row, col = edge_index
        
        # Get relative positional encodings
        rel_pos = x[row] - x[col]  # [n_edges, 3]
        rel_dist = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)  # [n_edges, 1]
        
        # Process edges
        edge_attr = torch.cat([h[row], h[col], rel_dist], dim=-1)  # [n_edges, input_nf * 2 + 1]
        edge_features = self.edge_mlp(edge_attr)  # [n_edges, hidden_nf]
        
        # Feature attention
        qkv = self.to_qkv(h).chunk(3, dim=-1)  # [n_nodes, n_heads * dim_head] each
        q, k, v = map(lambda t: t.reshape(n_nodes, self.n_heads, self.dim_head), qkv)
        
        # Compute attention scores - correct einsum to produce [n_nodes, n_heads, n_nodes]
        dots = torch.einsum('bhi,bji->bhj', q, k) * (self.dim_head ** -0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Fix mask expansion to match dots shape
            mask_expanded = mask.unsqueeze(1)  # [n_nodes, 1, n_nodes]
            dots = dots.masked_fill(~mask_expanded, -1e9)
        
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        feats = torch.einsum('bhj,bji->bhi', attn, v.transpose(1, 2))
        feats = feats.reshape(n_nodes, self.n_heads * self.dim_head)
        feats = self.to_out(feats)  # [n_nodes, output_nf]
        
        # Coordinate attention - create mask for edges if needed
        if mask is not None:
            # Create edge mask based on node mask
            edge_mask = mask[row, col].unsqueeze(-1)  # [n_edges, 1]
            edge_weights = self.coors_mlp(edge_features)  # [n_edges, 1]
            edge_weights = edge_weights.masked_fill(~edge_mask, -1e9)
        else:
            edge_weights = self.coors_mlp(edge_features)  # [n_edges, 1]
        
        coord_weights = F.softmax(edge_weights, dim=0)
        
        # Update coordinates while preserving E(n) equivariance
        coor_diff = coord_weights * rel_pos  # [n_edges, 3]
        coor_out = torch.zeros_like(x)  # [n_nodes, 3]
        coor_out.index_add_(0, row, coor_diff)
        
        return feats, coor_out

class EnTransformerLayer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.attention = EnAttention(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        # Add projection layer for residual connection if dimensions don't match
        self.proj = nn.Linear(input_nf, output_nf) if input_nf != output_nf else nn.Identity()
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(output_nf, hidden_nf * 2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, output_nf)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_nf)
        self.norm2 = nn.LayerNorm(output_nf)

    def forward(self, h, x, edge_index, mask=None):
        # Attention
        h_norm = self.norm1(h)
        h_attn, x_update = self.attention(h_norm, x, edge_index, mask=mask)
        h = self.proj(h) + h_attn  # Project h to match h_attn dimensions
        x = x + x_update  # Update coordinates
        
        # Feedforward
        h = h + self.ff(self.norm2(h))  # Residual connection
        
        return h, x

class EnTransformer(nn.Module):
    """
    Full E(n)-Transformer model with multiple layers and mask support.
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

    def forward(self, h, x, edge_index, mask=None):
        """
        h: Input node features [n_nodes, input_nf]
        x: Input coordinates [n_nodes, 3] 
        edge_index: Graph connectivity [2, n_edges]
        mask: None
        """
        for layer in self.layers:
            h, x = layer(h, x, edge_index, mask=mask)
        return h, x