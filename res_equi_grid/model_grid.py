import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add
import numpy as np

class EquivariantLayer(nn.Module):
    """
    Equivariant layer for learning orientations
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.message_net1 = nn.Sequential(
            nn.Linear(2*in_features+1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )
        self.message_net2 = nn.Sequential(
            nn.Linear(2*in_features+1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )
        self.p = 5  # Polynomial basis degree

    def forward(self, x, pos, edge_index):
        """
        Learn orientations for each point
        
        Args:
            x: Node features [N, F]
            pos: Node positions [N, 3]
            edge_index: Edge indices [2, E]
        """
        # Message passing
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
        vec1, vec2 = self.message(x[row], x[col], dist, pos[row], pos[col])
        
        # Aggregate messages
        vec1_out = scatter_add(vec1, col, dim=0, dim_size=x.size(0))
        vec2_out = scatter_add(vec2, col, dim=0, dim_size=x.size(0))
        
        # Apply Gram-Schmidt orthogonalization
        return self.gram_schmidt_batch(vec1_out, vec2_out)

    def gram_schmidt_batch(self, v1, v2):
        """
        Apply Gram-Schmidt to create orthogonal basis
        """
        # Normalize first vector
        n1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        
        # Make second vector orthogonal to first
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2 = n2_prime / (torch.norm(n2_prime, dim=-1, keepdim=True) + 1e-8)
        
        # Create third vector using cross product for right-handed system
        n3 = torch.cross(n1, n2, dim=-1)
        
        # Stack to create orientation matrices [N, 3, 3]
        return torch.stack([n1, n2, n3], dim=-2)
    
    def omega(self, dist):
        """Smoothing function for distance weighting"""
        r_max = 4.5  # Maximum radius
        out = 1 - (self.p+1)*(self.p+2)/2 * (dist/r_max)**self.p + \
              self.p*(self.p+2) * (dist/r_max)**(self.p+1) - \
              self.p*(self.p+1)/2 * (dist/r_max)**(self.p+2)
        return out
    
    def message(self, x_i, x_j, dist, pos_i, pos_j):
        """Create messages between nodes"""
        # Combine features and distance
        x_ij = torch.cat([x_i, x_j, dist], dim=-1)
        
        # Get message weights
        mes_1 = self.message_net1(x_ij)
        mes_2 = self.message_net2(x_ij)
        
        # Apply distance weighting
        coe = self.omega(dist)
        
        # Get normalized direction vector
        norm_vec = (pos_i - pos_j) / (torch.norm(pos_i - pos_j, dim=-1, keepdim=True) + 1e-8)
        
        return norm_vec * coe * mes_1, norm_vec * coe * mes_2

class GridificationLayer(nn.Module):
    """
    Layer for mapping point cloud to grid
    """
    def __init__(self, node_features, hidden_features):
        super().__init__()
        self.node_model = nn.Sequential(
            nn.Linear(node_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.edge_model = nn.Sequential(
            nn.Linear(6, hidden_features),  # 3D coordinates of both points
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.message_model = nn.Sequential(
            nn.Linear(2*hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.update_model = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
    def forward(self, node_features, node_pos, grid_pos, edge_index):
        """
        Map point cloud to grid
        
        Args:
            node_features: Point cloud features [N, F]
            node_pos: Point cloud positions [N, 3]
            grid_pos: Grid positions [G, 3]
            edge_index: Edge indices between point cloud and grid [2, E]
        """
        # Process node features
        node_features = self.node_model(node_features)
        
        # Generate messages
        messages = self.message(node_features, node_pos, grid_pos, edge_index)
        
        # Update grid features
        grid_features = self.update(messages, edge_index[1], grid_pos.size(0))
        
        return grid_features
        
    def message(self, node_features, node_pos, grid_pos, edge_index):
        """Generate messages between point cloud and grid"""
        source, target = edge_index
        
        # Get positions of connected points
        pos_source = node_pos[source]
        pos_target = grid_pos[target]
        
        # Create edge features from positions
        edge_attr = torch.cat([pos_source, pos_target], dim=-1)
        edge_features = self.edge_model(edge_attr)
        
        # Combine with node features
        node_features_source = node_features[source]
        message_input = torch.cat([node_features_source, edge_features], dim=-1)
        
        # Generate message
        message = self.message_model(message_input)
        
        return message
        
    def update(self, messages, target_indices, num_grid_points):
        """Update grid features based on messages"""
        # Count messages per grid point for normalization
        num_messages = torch.bincount(target_indices, minlength=num_grid_points).to(messages.device)
        num_messages = torch.clamp(num_messages, min=1).unsqueeze(-1)
        
        # Aggregate messages for each grid point
        grid_features = scatter_add(messages, target_indices, dim=0, dim_size=num_grid_points)
        
        # Normalize by number of messages
        grid_features = grid_features / num_messages
        
        # Apply update network
        grid_features = self.update_model(grid_features)
        
        return grid_features

class GridConvBlock(nn.Module):
    """3D convolutional block for processing grid data"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                     stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x):
        """Forward pass with residual connection"""
        return self.activation(self.conv(x) + self.shortcut(x))

class ResGridCNN(nn.Module):
    """Residual 3D CNN for processing grid data"""
    def __init__(self, in_channels, hidden_channels, num_classes, num_blocks=[2, 2, 2]):
        super().__init__()
        
        self.input_bn = nn.BatchNorm3d(in_channels)
        
        self.layer1 = self._make_layer(in_channels, hidden_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(hidden_channels, hidden_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(hidden_channels*2, hidden_channels*4, num_blocks[2], stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidden_channels*4, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block handles dimension change
        layers.append(GridConvBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(GridConvBlock(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Grid features [B, C, D, H, W]
        """
        x = self.input_bn(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class EquivariantGridModel(nn.Module):
    """
    Full model combining equivariant learning, gridification, and 3D CNN
    for residue identity prediction.
    """
    def __init__(self, 
                 in_node_nf=9,  # Number of atom types
                 hidden_nf=128, 
                 out_node_nf=20,  # Number of amino acid types
                 grid_size=9,
                 device='cuda'):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.grid_size = grid_size
        self.device = device
        
        # Initial node embedding
        self.atom_embedding = nn.Embedding(in_node_nf, hidden_nf)
        
        # Equivariant orientation learning
        self.equivariant_layer = EquivariantLayer(hidden_nf, hidden_nf)
        
        # Gridification layer
        self.gridification = GridificationLayer(hidden_nf, hidden_nf)
        
        # 3D CNN for grid processing
        self.grid_cnn = ResGridCNN(
            in_channels=hidden_nf,
            hidden_channels=hidden_nf//2,
            num_classes=out_node_nf
        )
        
    def forward(self, atoms, x, edge_index, data):
        """
        Forward pass
        
        Args:
            atoms: Atom types [N]
            x: Atom coordinates [N, 3]
            edge_index: Edge indices [2, E]
            data: Additional data including grid information
        """
        batch_size = data.ptr.size(0) - 1
        
        # Embed atoms
        h = self.atom_embedding(atoms)
        
        # Learn orientations
        orientations = self.equivariant_layer(h, x, edge_index)
        
        # Project grid positions into local frame
        grid_pos = data.grid_coords
        grid_edge_index = data.grid_edge_index
        
        # Apply gridification to map point cloud to grid
        grid_features = self.gridification(h, x, grid_pos, grid_edge_index)
        
        # Reshape grid features for CNN processing [B, C, D, H, W]
        grid_features = grid_features.reshape(
            batch_size, 
            self.grid_size, self.grid_size, self.grid_size, 
            self.hidden_nf
        ).permute(0, 4, 1, 2, 3)
        
        # Process grid with 3D CNN
        logits = self.grid_cnn(grid_features)
        
        return logits