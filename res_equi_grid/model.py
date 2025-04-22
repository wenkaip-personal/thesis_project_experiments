import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean, scatter_max

class OrientedMP(nn.Module):
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
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*hidden_nf, out_node_nf)
        )
    
    def forward(self, h, x, edge_index, batch):
        # Embed node features
        h = self.atom_embedding(h)
        
        # Learn orientations for each atom
        orientations = self.orientation_network(h, x, edge_index, batch.edge_s)
        
        # Message passing with learned orientations
        for mp_layer in self.mp_layers:
            h = mp_layer(h, x, edge_index, orientations, batch.edge_s)
        
        # Get prediction for the central atom (CA)
        return self.mlp(h[batch.ca_idx + batch.ptr[:-1]])
    
class OrientationLearner(nn.Module):
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
        
        # MLPs for generating two vectors for each node
        self.vec1_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3)
        )
        
        self.vec2_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3)
        )
    
    def forward(self, h, x, edge_index, edge_attr):
        row, col = edge_index
        
        # Create edge features
        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        edge_features = self.edge_mlp(edge_features)
        
        # Aggregate edge features to nodes
        node_features = scatter_mean(edge_features, row, dim=0, dim_size=h.size(0))
        
        # Generate two vectors for each node
        vec1 = self.vec1_mlp(node_features)
        vec2 = self.vec2_mlp(node_features)
        
        # Normalize first vector
        vec1_norm = F.normalize(vec1, p=2, dim=-1)
        
        # Make second vector orthogonal to the first
        vec2 = vec2 - torch.sum(vec2 * vec1_norm, dim=-1, keepdim=True) * vec1_norm
        vec2_norm = F.normalize(vec2, p=2, dim=-1)
        
        # Create third vector using cross product
        vec3 = torch.cross(vec1_norm, vec2_norm, dim=-1)
        
        # Stack vectors to create orientation matrices
        orientations = torch.stack([vec1_norm, vec2_norm, vec3], dim=-1)
        
        return orientations
    
class OrientedMessagePassing(MessagePassing):
    def __init__(self, hidden_nf, edge_nf):
        super().__init__(aggr='mean')
        self.hidden_nf = hidden_nf
        
        # MLP for processing messages
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2 + edge_nf + 3, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # MLP for updating node features
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
    
    def forward(self, h, x, edge_index, orientations, edge_attr):
        # Manually lift orientations to edges
        row, col = edge_index
        orientations_i = orientations[row]  # Shape: [num_edges, 3, 3]
        
        # Propagate messages with pre-lifted orientations
        return self.propagate(edge_index, h=h, x=x, orientations_i=orientations_i, edge_attr=edge_attr)
    
    def message(self, h_i, h_j, x_i, x_j, orientations_i, edge_attr):
        # Calculate relative position
        rel_pos = x_j - x_i
        
        # Project relative position using orientation of node i
        # orientations_i shape: [num_edges, 3, 3]
        # rel_pos shape: [num_edges, 3]
        projected_rel_pos = torch.bmm(orientations_i.transpose(1, 2), rel_pos.unsqueeze(-1)).squeeze(-1)
        
        # Construct message features
        message_features = torch.cat([h_i, h_j, edge_attr, projected_rel_pos], dim=-1)
        
        # Process message
        return self.message_mlp(message_features)
    
    def update(self, aggr_out, h):
        # Combine aggregated messages with node features
        combined = torch.cat([h, aggr_out], dim=-1)
        
        # Update node features
        return self.update_mlp(combined)