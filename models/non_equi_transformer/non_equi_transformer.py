import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class EnAttention(nn.Module):
    """
    Modified E(n) Attention Layer with equivariance removed through minimal changes.
    The structure remains identical to the original, but absolute positions are incorporated.
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
        
        # 2. Relative Position Processing - unchanged structure
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, hidden_nf // 2),
            nn.SiLU(),
            nn.Linear(hidden_nf // 2, n_heads)
        )
        
        # 3. Edge Feature Processing - MODIFIED to include absolute positions
        # Changed from input_nf * 2 + 1 to input_nf * 2 + 1 + 6 (adding 3D coords for both nodes)
        edge_input_dim = input_nf * 2 + 1 + 6  # features from i, j nodes + distance + absolute positions
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Edge weights for attention - unchanged
        self.edge_weight = nn.Linear(hidden_nf, n_heads)
        
        # 5. Coordinate Update Mechanism - MODIFIED input dimension
        # Changed to include absolute positions in the message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_nf + 1 + 3 + 6, hidden_nf),  # edge features + attention + relative coords + absolute coords
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Rest remains unchanged
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_nf, n_heads),
            nn.Sigmoid()
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, n_heads)
        )
        
        # Learnable parameters for coordinate combination - unchanged
        self.coord_weights = nn.Parameter(torch.ones(n_heads))
        
        # 7. Cross-Product Enhancement (Optional) - unchanged
        self.use_cross_product = True
        if self.use_cross_product:
            self.cross_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, n_heads)
            )

    def forward(self, h, x, edge_index, mask=None):
        """
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3]
        edge_index: Graph connectivity [2, n_edges]
        mask: Optional mask tensor [n_edges] - True for valid edges, False for invalid ones
        """
        row, col = edge_index
        n_nodes = h.shape[0]
        
        # 1. Input Transformation and Initial Projections - unchanged
        q = self.to_q(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        k = self.to_k(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        v = self.to_v(h).view(-1, self.n_heads, self.dim_head)  # [n_nodes, n_heads, dim_head]
        
        # 2. Relative Position Processing - unchanged
        rel_pos = x[row] - x[col]  # [n_edges, 3]
        rel_dist = torch.sum(rel_pos**2, dim=-1, keepdim=True)  # [n_edges, 1]
        pos_encoding = self.pos_mlp(rel_dist)  # [n_edges, n_heads]
        
        # 3. Edge Feature Processing - MODIFIED to include absolute positions
        # Normalize absolute positions to prevent gradient issues
        x_normalized = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)
        edge_attr = torch.cat([
            h[row], h[col], 
            rel_dist,
            x_normalized[row], x_normalized[col]  # ADDED: normalized absolute positions
        ], dim=-1)  # [n_edges, input_nf*2+1+6]
        edge_features = self.edge_mlp(edge_attr)  # [n_edges, hidden_nf]
        edge_weights = self.edge_weight(edge_features)  # [n_edges, n_heads]
        
        # 4. Attention Score Computation - unchanged
        q_i = q[row]  # [n_edges, n_heads, dim_head]
        k_j = k[col]  # [n_edges, n_heads, dim_head]
        
        # Dot product attention score (equation 8)
        attn_score = torch.sum(q_i * k_j, dim=-1) / (self.dim_head**0.5)  # [n_edges, n_heads]
        attn_score = attn_score + pos_encoding + edge_weights  # Add positional and edge information
        
        # Apply mask to attention scores if provided - unchanged
        if mask is not None:
            attn_score = torch.where(
                mask.unsqueeze(-1),  # Expand mask to [n_edges, n_heads]
                attn_score,
                torch.tensor(-1e9, device=attn_score.device)
            )
        
        # Apply softmax per source node and head (equation 9) - unchanged
        attn_score_max, _ = torch_scatter.scatter_max(attn_score, row, dim=0)
        attn_score = attn_score - attn_score_max[row]
        attn_score = torch.exp(attn_score)
        attn_sum = torch_scatter.scatter_add(attn_score, row, dim=0)
        attn_score = attn_score / (attn_sum[row] + 1e-8)  # [n_edges, n_heads]
        
        # 5. Coordinate Update Mechanism - MODIFIED to include absolute positions
        # Create message from attention, edge features, relative positions, and absolute positions
        msg_input = torch.cat([
            edge_features,  # [n_edges, hidden_nf]
            attn_score.mean(dim=1, keepdim=True),  # [n_edges, 1]
            rel_pos,  # [n_edges, 3]
            x_normalized[row], x_normalized[col]  # ADDED: normalized absolute positions
        ], dim=-1)
        
        messages = self.msg_mlp(msg_input)  # [n_edges, hidden_nf]
        
        # Gating mechanism (equation 11) - unchanged
        gates = self.gate_mlp(messages)  # [n_edges, n_heads]
        
        # Coordinate weights (equation 12) - unchanged
        coord_weights = self.coord_mlp(messages)  # [n_edges, n_heads]
        
        # Coordinate update (equation 13) - MODIFIED to break equivariance
        coord_update = torch.zeros_like(x)  # [n_nodes, 3]
        
        for h_idx in range(self.n_heads):
            # For each head, compute its contribution
            head_weight = gates[:, h_idx:h_idx+1] * coord_weights[:, h_idx:h_idx+1] * self.coord_weights[h_idx]
            # MODIFIED: Use a weighted combination of relative positions and absolute target positions
            # This breaks equivariance by making updates depend on absolute positions
            weighted_pos = head_weight * (0.9 * rel_pos + 0.1 * x[col])
            
            # Sum over neighbors (using scatter_add)
            coord_update.index_add_(0, row, weighted_pos / (n_nodes - 1))
        
        # 7. Cross-Product Enhancement (Optional) - unchanged mechanism
        if self.use_cross_product:
            # Compute cross products between relative positions
            cross_vectors = torch.cross(rel_pos, rel_pos.roll(1, 0))  # [n_edges, 3]
            cross_gates = self.cross_mlp(messages)  # [n_edges, n_heads]
            
            cross_update = torch.zeros_like(x)  # [n_nodes, 3]
            for h_idx in range(self.n_heads):
                weighted_cross = cross_gates[:, h_idx:h_idx+1] * cross_vectors
                cross_update.index_add_(0, row, weighted_cross / (n_nodes - 1))
            
            # Combine with regular update (equation 20)
            coord_update = coord_update + cross_update
        
        # 6. Feature Update Mechanism - unchanged
        v_j = v[col]  # [n_edges, n_heads, dim_head]
        
        # For each head, apply attention to values
        weighted_values = torch.zeros(n_nodes, self.n_heads, self.dim_head, device=h.device)
        
        for h_idx in range(self.n_heads):
            head_attn = attn_score[:, h_idx:h_idx+1]
            head_values = v_j[:, h_idx]  # [n_edges, dim_head]
            weighted_head_values = head_attn * head_values
            weighted_values[:, h_idx].index_add_(0, row, weighted_head_values)
        
        # Combine heads (equation 15)
        output_features = weighted_values.reshape(n_nodes, -1)  # [n_nodes, n_heads*dim_head]
        output_features = self.to_out(output_features)  # [n_nodes, output_nf]
        
        return output_features, coord_update


class EnTransformerLayer(nn.Module):
    """
    Transformer layer - structure remains completely unchanged
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_heads=4, dim_head=64):
        super().__init__()
        self.attention = EnAttention(input_nf, output_nf, hidden_nf, n_heads, dim_head)
        
        # Add projection layer for residual connection if dimensions don't match
        self.proj = nn.Linear(input_nf, output_nf) if input_nf != output_nf else nn.Identity()
        
        # Feedforward network (equation 21)
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
        # Attention with pre-normalization (equation 16)
        h_norm = self.norm1(h)
        h_attn, x_update = self.attention(h_norm, x, edge_index, mask=mask)
        h = self.proj(h) + self.dropout(h_attn)  # Residual connection with projection
        
        # Update coordinates
        x = x + x_update
        
        # Feedforward with residual connection (equation 21)
        h = h + self.ff(h)
        
        return h, x


class EnTransformer(nn.Module):
    """
    Full E(n)-Transformer model - structure remains completely unchanged
    """
    def __init__(self, input_nf, output_nf, hidden_nf, n_layers=4, n_heads=4, dim_head=64):
        super().__init__()
        # Input embedding
        self.input_embedding = nn.Linear(input_nf, hidden_nf)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnTransformerLayer(
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
            EnTransformerLayer(
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
        x: Input coordinates [n_nodes, 3] 
        edge_index: Graph connectivity [2, n_edges]
        mask: Optional mask tensor for edges [n_edges]
        """
        # Initial embedding
        h = self.input_embedding(h)
        
        # Process through layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index, mask=mask)
            
        return h, x