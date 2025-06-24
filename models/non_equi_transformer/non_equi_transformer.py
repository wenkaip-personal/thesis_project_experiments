import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class NonEquivariantAttention(nn.Module):
    """
    Non-Equivariant Attention Layer that processes coordinates and features
    without preserving E(n) equivariance. Modified from EnAttention.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_heads = n_heads
        self.dim_head = dim_head
        
        # 1. Input Transformation and Initial Projections (Q, K, V) - unchanged
        self.to_q = nn.Linear(input_nf, n_heads * dim_head)
        self.to_k = nn.Linear(input_nf, n_heads * dim_head)
        self.to_v = nn.Linear(input_nf, n_heads * dim_head)
        
        # Output projection - unchanged
        self.to_out = nn.Linear(n_heads * dim_head, output_nf)
        
        # 2. Position Processing - Modified to use absolute positions
        # Now takes both source and target absolute positions
        self.pos_mlp = nn.Sequential(
            nn.Linear(6, hidden_nf),  # 3D coords for source + 3D coords for target
            nn.SiLU(),
            nn.Linear(hidden_nf, n_heads)
        )
        
        # 3. Edge Feature Processing - Modified to include absolute positions
        edge_input_dim = input_nf * 2 + 7  # features from i, j nodes + distance + 6D absolute coords
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Edge weights for attention - unchanged
        self.edge_weight = nn.Linear(hidden_nf, n_heads)
        
        # 5. Coordinate Processing - Completely redesigned for non-equivariance
        # Instead of updating coordinates, we learn position-dependent features
        self.coord_feature_mlp = nn.Sequential(
            nn.Linear(3, hidden_nf),  # Takes absolute coordinates
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf)
        )
        
        # Remove coordinate update mechanisms and cross-product enhancement
        # No coord_mlp, gate_mlp, coord_weights, or cross_mlp

    def forward(self, h, x, edge_index, mask=None):
        """
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3] - treated as features, not updated
        edge_index: Graph connectivity [2, n_edges]
        mask: Optional mask tensor [n_edges] - True for valid edges, False for invalid ones
        """
        row, col = edge_index
        n_nodes = h.shape[0]
        
        # 1. Input Transformation and Initial Projections
        q = self.to_q(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        k = self.to_k(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        v = self.to_v(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        
        # 2. Position Processing - Using absolute positions
        # Concatenate source and target absolute positions
        abs_pos_pairs = torch.cat([x[row], x[col]], dim=-1)  # [n_edges, 6]
        pos_encoding = self.pos_mlp(abs_pos_pairs)  # [n_edges, n_heads]
        
        # Also compute relative positions for distance (but not for equivariance)
        rel_pos = x[row] - x[col]  # [n_edges, 3]
        rel_dist = torch.sum(rel_pos**2, dim=-1, keepdim=True)  # [n_edges, 1]
        
        # 3. Edge Feature Processing - Including absolute positions
        edge_attr = torch.cat([
            h[row], h[col],  # Node features
            rel_dist,        # Distance
            x[row], x[col]   # Absolute positions
        ], dim=-1)  # [n_edges, input_nf*2+1+6]
        
        edge_features = self.edge_mlp(edge_attr)  # [n_edges, hidden_nf]
        edge_weights = self.edge_weight(edge_features)  # [n_edges, n_heads]
        
        # 4. Attention Score Computation - unchanged mechanism
        q_i = q[row]  # [n_edges, n_heads, dim_head]
        k_j = k[col]  # [n_edges, n_heads, dim_head]
        
        # Dot product attention score
        attn_score = torch.sum(q_i * k_j, dim=-1) / (self.dim_head**0.5)  # [n_edges, n_heads]
        attn_score = attn_score + pos_encoding + edge_weights  # Add positional and edge information
        
        # Apply mask to attention scores if provided
        if mask is not None:
            attn_score = torch.where(
                mask.unsqueeze(-1),  # Expand mask to [n_edges, n_heads]
                attn_score,
                torch.tensor(-1e9, device=attn_score.device)
            )
        
        # Apply softmax per source node and head
        attn_score_max, _ = torch_scatter.scatter_max(attn_score, row, dim=0)
        attn_score = attn_score - attn_score_max[row]
        attn_score = torch.exp(attn_score)
        attn_sum = torch_scatter.scatter_add(attn_score, row, dim=0)
        attn_score = attn_score / (attn_sum[row] + 1e-8)  # [n_edges, n_heads]
        
        # 5. Feature Update Mechanism - unchanged
        v_j = v[col]  # [n_edges, n_heads, dim_head]
        
        # For each head, apply attention to values
        weighted_values = torch.zeros(n_nodes, self.n_heads, self.dim_head, device=h.device)
        
        for h_idx in range(self.n_heads):
            head_attn = attn_score[:, h_idx:h_idx+1]
            head_values = v_j[:, h_idx]  # [n_edges, dim_head]
            weighted_head_values = head_attn * head_values
            weighted_values[:, h_idx].index_add_(0, row, weighted_head_values)
        
        # Combine heads
        output_features = weighted_values.reshape(n_nodes, -1)  # [n_nodes, n_heads*dim_head]
        output_features = self.to_out(output_features)  # [n_nodes, output_nf]
        
        # 6. Position-dependent features (non-equivariant)
        # Learn features from absolute coordinates
        coord_features = self.coord_feature_mlp(x)  # [n_nodes, output_nf]
        
        # Combine attention output with position-dependent features
        output_features = output_features + coord_features
        
        # Return features and unchanged coordinates
        # No coordinate update - coordinates remain the same
        return output_features, torch.zeros_like(x)  # Zero update to maintain interface


class NonEquivariantTransformerLayer(nn.Module):
    """
    Non-Equivariant Transformer layer that processes but doesn't update coordinates
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.attention = NonEquivariantAttention(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        # Add projection layer for residual connection if dimensions don't match
        self.proj = nn.Linear(input_nf, output_nf) if input_nf != output_nf else nn.Identity()
        
        # Feedforward network - unchanged
        self.ff = nn.Sequential(
            nn.LayerNorm(output_nf),
            nn.Linear(output_nf, hidden_nf * 2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, output_nf)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_nf)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h, x, edge_index, mask=None):
        # Attention with pre-normalization
        h_norm = self.norm1(h)
        h_attn, x_update = self.attention(h_norm, x, edge_index, mask=mask)
        h = self.proj(h) + self.dropout(h_attn)  # Residual connection with projection
        
        # Coordinates are not updated in non-equivariant version
        # x = x + x_update  # This line is effectively x = x + 0
        
        # Feedforward with residual connection
        h = h + self.ff(h)
        
        return h, x


class NonEquivariantTransformer(nn.Module):
    """
    Full Non-Equivariant Transformer model with multiple layers.
    Processes spatial information without maintaining E(n) equivariance.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_layers=4, n_heads=4, dim_head=64):
        super().__init__()
        # Input embedding
        self.input_embedding = nn.Linear(input_nf, hidden_nf)
        
        # Additional spatial encoding network (non-equivariant)
        # This processes absolute coordinates to create position-aware features
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NonEquivariantTransformerLayer(
                hidden_nf,
                hidden_nf,
                hidden_nf,
                n_heads,
                dim_head
            )
            for i in range(n_layers - 1)
        ])
        
        # Final layer with output size
        self.layers.append(
            NonEquivariantTransformerLayer(
                hidden_nf,
                output_nf,
                hidden_nf,
                n_heads,
                dim_head
            )
        )

    def forward(self, h, x, edge_index, mask=None):
        """
        h: Input node features [n_nodes, input_nf]
        x: Input coordinates [n_nodes, 3] - treated as features, not updated
        edge_index: Graph connectivity [2, n_edges]
        mask: Optional mask tensor for edges [n_edges]
        """
        # Initial embedding
        h = self.input_embedding(h)
        
        # Add spatial encoding to features (non-equivariant)
        spatial_features = self.spatial_encoder(x)
        h = h + spatial_features
        
        # Process through layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index, mask=mask)
            
        return h, x