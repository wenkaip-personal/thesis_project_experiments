import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean, scatter_max

# Small constant to prevent numerical issues
EPS = 1e-8

class ResOrientedMP(nn.Module):
    """
    Residue Oriented Message Passing model with equivariance properties.
    """
    def __init__(self, in_node_nf=9, hidden_nf=128, out_node_nf=20, in_edge_nf=16, n_layers=4):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.n_layers = n_layers
        
        # Node feature embedding
        self.atom_embedding = nn.Embedding(in_node_nf, hidden_nf)
        
        # Orientation learning module
        self.orientation_network = OrientationLearner(
            hidden_nf=hidden_nf,
            edge_nf=in_edge_nf
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            OrientedMessagePassing(
                hidden_nf=hidden_nf,
                edge_nf=in_edge_nf
            ) for _ in range(n_layers)
        ])
        
        # Final prediction MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, 2*hidden_nf),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2*hidden_nf, out_node_nf)
        )
    
    def forward(self, h, x, edge_index, batch):
        # Embed node features
        h = self.atom_embedding(h)
        
        # Add small noise to coordinates for better stability
        x = x + torch.randn_like(x) * 0.01
        
        # Center coordinates per graph to achieve translation invariance
        if hasattr(batch, 'batch'):
            batch_idx = batch.batch
            x = x - scatter_mean(x, batch_idx, dim=0)[batch_idx]
        
        # Learn orientations for each atom
        orientations = self.orientation_network(h, x, edge_index, batch.edge_s)
        
        # Message passing with learned orientations
        for mp_layer in self.mp_layers:
            h = mp_layer(h, x, edge_index, orientations, batch.edge_s)
        
        # Get prediction for the central atom (CA)
        if hasattr(batch, 'ca_idx') and hasattr(batch, 'ptr'):
            return self.mlp(h[batch.ca_idx + batch.ptr[:-1]])
        else:
            return self.mlp(h)


class OrientationLearner(nn.Module):
    """
    Orientation learning module.
    """
    def __init__(self, hidden_nf, edge_nf):
        super().__init__()
        self.hidden_nf = hidden_nf
        
        # MLP for processing edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_nf + hidden_nf*2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
        )
        
        # MLPs for generating two vector components for orientation
        self.vec1_net = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
        self.vec2_net = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
    
    def forward(self, h, x, edge_index, edge_attr):
        row, col = edge_index
        
        # Get relative positions
        rel_pos = x[col] - x[row]
        
        # Create edge features
        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        edge_features = self.edge_mlp(edge_features)
        
        # Weight the relative positions to create basis vectors
        vec1_weights = self.vec1_net(edge_features)
        vec2_weights = self.vec2_net(edge_features)
        
        vec1 = rel_pos * vec1_weights
        vec2 = rel_pos * vec2_weights
        
        # Aggregate weighted vectors to nodes
        vec1_per_node = scatter_add(vec1, row, dim=0, dim_size=h.size(0))
        vec2_per_node = scatter_add(vec2, row, dim=0, dim_size=h.size(0))
        
        # Apply Gram-Schmidt process to create orthogonal basis
        return self.gram_schmidt(vec1_per_node, vec2_per_node)
    
    def gram_schmidt(self, v1, v2):
        """
        Apply Gram-Schmidt process to create orthogonal basis vectors
        """
        # Normalize first vector
        e1 = F.normalize(v1, p=2, dim=-1)
        
        # Make second vector orthogonal to first
        v2_proj = v2 - torch.sum(e1 * v2, dim=-1, keepdim=True) * e1
        e2 = F.normalize(v2_proj, p=2, dim=-1)
        
        # Create third vector using cross product for right-handed system
        e3 = torch.cross(e1, e2, dim=-1)
        
        # Stack to create orientation matrices [batch_size, 3, 3]
        return torch.stack([e1, e2, e3], dim=-1)


class OrientedMessagePassing(MessagePassing):
    """
    Message passing with orientations.
    """
    def __init__(self, hidden_nf, edge_nf):
        super().__init__(aggr='add')  # Use 'add' aggregation
        self.hidden_nf = hidden_nf
        
        # MLP for processing messages with distance encoding
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2 + edge_nf + 3, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Attention mechanism for weighting messages
        self.attention = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid()
        )
        
        # MLP for updating node features
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
    
    def forward(self, h, x, edge_index, orientations, edge_attr):
        # Propagate messages
        return self.propagate(
            edge_index, 
            h=h, 
            x=x,
            orientations=orientations,
            edge_attr=edge_attr
        )
    
    def message(self, h_i, h_j, x_i, x_j, orientations_i, edge_attr):
        # Get relative positions
        rel_pos = x_j - x_i
        
        # Project relative positions to local coordinate frame
        projected_rel_pos = torch.bmm(
            orientations_i.transpose(1, 2), 
            rel_pos.unsqueeze(-1)
        ).squeeze(-1)
        
        # Combine features for message
        message_input = torch.cat([h_i, h_j, edge_attr, projected_rel_pos], dim=-1)
        
        # Process message
        message = self.message_mlp(message_input)
        
        # Apply attention mechanism
        message = message * self.attention(message)
        
        return message
    
    def update(self, aggr_out, h):
        # Combine aggregated messages with node features
        combined = torch.cat([h, aggr_out], dim=-1)
        
        # Update node features with residual connection
        return h + self.update_mlp(combined)