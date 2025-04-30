import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_add

EPS = 1e-8

class EquivariantResModel(nn.Module):
    def __init__(self, in_node_nf=9, hidden_nf=128, out_node_nf=20, edge_nf=16, n_layers=4, device='cuda'):
        """
        Equivariant model for residue identity prediction.
        
        Args:
            in_node_nf: Input dimension of node features (number of atom types)
            hidden_nf: Hidden dimension
            out_node_nf: Output dimension (number of amino acid classes)
            edge_nf: Edge feature dimension
            n_layers: Number of message passing layers
            device: Device to run the model on
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        
        # Node embedding
        self.atom_embedding = nn.Embedding(in_node_nf, hidden_nf)
        
        # Orientation network
        self.orientation_mlp1 = nn.Sequential(
            nn.Linear(hidden_nf*2 + edge_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
        self.orientation_mlp2 = nn.Sequential(
            nn.Linear(hidden_nf*2 + edge_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        for i in range(n_layers):
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_nf*2 + edge_nf + 3, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf)
            ))
            
        self.update_layers = nn.ModuleList()
        for i in range(n_layers):
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_nf*2, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf)
            ))
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf*2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_nf*2, out_node_nf)
        )
        
    def get_orientations(self, h, x, edge_index, edge_attr):
        """Learn orientations for each node"""
        row, col = edge_index
        
        # Create message input features
        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        
        # Get weighted vectors from relative positions
        rel_pos = x[row] - x[col]
        
        # Weight the relative positions to form basis vectors
        vec1_weights = self.orientation_mlp1(edge_features)
        vec2_weights = self.orientation_mlp2(edge_features)
        
        vec1 = rel_pos * vec1_weights
        vec2 = rel_pos * vec2_weights
        
        # Aggregate to get per-node vectors
        vec1_per_node = scatter_add(vec1, col, dim=0, dim_size=h.size(0))
        vec2_per_node = scatter_add(vec2, col, dim=0, dim_size=h.size(0))
        
        # Gram-Schmidt orthogonalization to create orientations
        return self.gram_schmidt_batch(vec1_per_node, vec2_per_node)
    
    def gram_schmidt_batch(self, v1, v2):
        """Apply Gram-Schmidt to create orthogonal basis"""
        # Normalize first vector
        v1_norm = torch.norm(v1, dim=-1, keepdim=True) + EPS
        e1 = v1 / v1_norm
        
        # Make second vector orthogonal to first
        proj = torch.sum(e1 * v2, dim=-1, keepdim=True) * e1
        v2_orthogonal = v2 - proj
        
        # Normalize second vector
        v2_norm = torch.norm(v2_orthogonal, dim=-1, keepdim=True) + EPS
        e2 = v2_orthogonal / v2_norm
        
        # Create third vector using cross product for right-handed system
        e3 = torch.cross(e1, e2)
        
        # Stack to create orientation matrices [batch_size, 3, 3]
        return torch.stack([e1, e2, e3], dim=-1)
    
    def message_passing(self, h, x, edge_index, orientations, edge_attr):
        """Message passing with learned orientations"""
        row, col = edge_index
        
        for i in range(self.n_layers):
            # Create messages with projected coordinates
            rel_pos = x[col] - x[row]
            
            # Project relative positions to local frame
            orientations_row = orientations[row]
            projected_pos = torch.bmm(
                orientations_row.transpose(1, 2), 
                rel_pos.unsqueeze(-1)
            ).squeeze(-1)
            
            # Compose messages
            message_input = torch.cat([h[row], h[col], edge_attr, projected_pos], dim=-1)
            messages = self.message_layers[i](message_input)
            
            # Aggregate messages
            aggregated = scatter_add(messages, row, dim=0, dim_size=h.size(0))
            
            # Update node features
            update_input = torch.cat([h, aggregated], dim=-1)
            h = h + self.update_layers[i](update_input)
            
        return h
    
    def forward(self, atoms, x, edge_index, data):
        """Forward pass"""
        # Embed atom features
        h = self.atom_embedding(atoms)
        
        # Center point cloud (translation invariance)
        if hasattr(data, 'batch'):
            batch_idx = data.batch
            x = x - global_mean_pool(x, batch_idx)[batch_idx]
        
        # Learn orientations
        orientations = self.get_orientations(h, x, edge_index, data.edge_s)
        
        # Message passing
        h = self.message_passing(h, x, edge_index, orientations, data.edge_s)
        
        # Get prediction for the central alpha carbon
        if hasattr(data, 'ca_idx') and hasattr(data, 'batch'):
            if hasattr(data, 'ptr'):
                # For batched data with pointer
                output = self.output_mlp(h[data.ca_idx + data.ptr[:-1]])
            else:
                # For single graphs
                output = self.output_mlp(h[data.ca_idx])
        else:
            # Fallback to all nodes
            output = self.output_mlp(h)
            
        return output