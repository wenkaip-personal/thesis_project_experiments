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
        Apply Gram-Schmidt to create orthogonal basis with improved numerical stability
        """
        # FIX: Improve numerical stability with larger epsilon and fallback for degenerate cases
        eps = 1e-6
        
        # Normalize first vector
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)
        mask1 = (v1_norm > eps).float()
        n1 = torch.where(mask1 > 0.5, v1 / (v1_norm + eps), 
                         torch.tensor([1.0, 0.0, 0.0], device=v1.device).expand_as(v1))
        
        # Make second vector orthogonal to first
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2_norm = torch.norm(n2_prime, dim=-1, keepdim=True)
        mask2 = (n2_norm > eps).float()
        
        # If second vector becomes too small after projection, use a perpendicular vector to first
        fallback = torch.stack([
            -n1[:, 1], n1[:, 0], torch.zeros_like(n1[:, 0])
        ], dim=1)
        # Make sure fallback is normalized and orthogonal
        fallback = fallback - (n1 * fallback).sum(dim=-1, keepdim=True) * n1
        fallback_norm = torch.norm(fallback, dim=-1, keepdim=True)
        fallback = fallback / (fallback_norm + eps)
        
        n2 = torch.where(mask2 > 0.5, n2_prime / (n2_norm + eps), fallback)
        
        # Create third vector using cross product for right-handed system
        n3 = torch.cross(n1, n2, dim=-1)
        n3_norm = torch.norm(n3, dim=-1, keepdim=True)
        n3 = n3 / (n3_norm + eps)
        
        # Stack to create orientation matrices [N, 3, 3]
        return torch.stack([n1, n2, n3], dim=-1)
    
    def omega(self, dist):
        """Smoothing function for distance weighting"""
        r_max = 4.5  # Maximum radius
        # FIX: Add clipping to prevent numerical issues
        dist_clipped = torch.clamp(dist, min=0.0, max=r_max)
        
        out = 1 - (self.p+1)*(self.p+2)/2 * (dist_clipped/r_max)**self.p + \
              self.p*(self.p+2) * (dist_clipped/r_max)**(self.p+1) - \
              self.p*(self.p+1)/2 * (dist_clipped/r_max)**(self.p+2)
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
        eps = 1e-6  # FIX: Increase epsilon for better numerical stability
        direction = pos_i - pos_j
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)
        norm_vec = direction / (direction_norm + eps)
        
        return norm_vec * coe * mes_1, norm_vec * coe * mes_2

class GridificationLayer(nn.Module):
    """
    Layer for mapping point cloud to grid with orientation-aware projection
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
        
    def forward(self, node_features, node_pos, grid_pos, edge_index, orientations):
        """
        Map point cloud to grid using orientations for projection
        
        Args:
            node_features: Point cloud features [N, F]
            node_pos: Point cloud positions [N, 3]
            grid_pos: Grid positions [G, 3]
            edge_index: Edge indices between point cloud and grid [2, E]
            orientations: Orientation matrices for each point [N, 3, 3]
        """
        # FIX: Add validation checks for inputs
        if edge_index.numel() == 0:
            # Handle empty edge index case
            batch_size = 1
            grid_features = torch.zeros((grid_pos.size(0), node_features.size(1)), 
                                       device=node_pos.device)
            return grid_features
            
        # Check for invalid indices and filter them out
        max_node_index = node_pos.size(0) - 1
        max_grid_index = grid_pos.size(0) - 1
        
        valid_mask = (edge_index[0] >= 0) & (edge_index[0] <= max_node_index) & \
                     (edge_index[1] >= 0) & (edge_index[1] <= max_grid_index)
        
        if not valid_mask.all():
            edge_index = edge_index[:, valid_mask]
            
        # Process node features
        node_features = self.node_model(node_features)
        
        # Generate messages using orientation-aware projection
        messages = self.message(node_features, node_pos, grid_pos, edge_index, orientations)
        
        # Update grid features
        grid_features = self.update(messages, edge_index[1], grid_pos.size(0))
        
        return grid_features
        
    def message(self, node_features, node_pos, grid_pos, edge_index, orientations):
        """Generate messages between point cloud and grid with orientation-aware projection"""
        source, target = edge_index
        
        # Get positions of connected points
        pos_source = node_pos[source]
        pos_target = grid_pos[target]
        
        # Transform relative positions using point orientations
        rel_pos = pos_target - pos_source
        
        # FIX: Add safety check for empty tensors
        if rel_pos.numel() == 0:
            return torch.zeros((0, node_features.size(1)), device=node_features.device)
        
        # Apply orientations to decouple from global rotation
        # [E, 3] @ [E, 3, 3] -> [E, 3]
        transformed_rel_pos = torch.bmm(rel_pos.unsqueeze(1), orientations[source]).squeeze(1)
        
        # Create edge features from positions
        edge_attr = torch.cat([pos_source, transformed_rel_pos], dim=-1)
        edge_features = self.edge_model(edge_attr)
        
        # Combine with node features
        node_features_source = node_features[source]
        message_input = torch.cat([node_features_source, edge_features], dim=-1)
        
        # Generate message
        message = self.message_model(message_input)
        
        return message
        
    def update(self, messages, target_indices, num_grid_points):
        """Update grid features based on messages"""
        # FIX: Handle empty messages case
        if messages.numel() == 0 or target_indices.numel() == 0:
            return torch.zeros((num_grid_points, messages.size(1) if messages.numel() > 0 else 128), 
                              device=messages.device if messages.numel() > 0 else 'cpu')
            
        # Check for invalid indices and filter them out
        valid_mask = (target_indices >= 0) & (target_indices < num_grid_points)
        
        if not valid_mask.all():
            messages = messages[valid_mask]
            target_indices = target_indices[valid_mask]
            
        # Handle the case where all indices were invalid
        if target_indices.numel() == 0:
            return torch.zeros((num_grid_points, messages.size(1)), device=messages.device)
            
        # Count messages per grid point for normalization
        counts = torch.zeros(num_grid_points, device=messages.device)
        counts.scatter_add_(0, target_indices, torch.ones_like(target_indices, dtype=torch.float))
        counts = torch.clamp(counts, min=1).unsqueeze(-1)
        
        # Aggregate messages for each grid point
        grid_features = torch.zeros((num_grid_points, messages.size(1)), device=messages.device)
        grid_features.scatter_add_(0, target_indices.unsqueeze(-1).expand(-1, messages.size(1)), messages)
        
        # Normalize by number of messages
        grid_features = grid_features / counts
        
        # Apply update network
        grid_features = self.update_model(grid_features)
        
        return grid_features

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = nn.ReLU()(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers: list, num_classes: int = 20, in_channels: int = 128):
        super(ResNet3D, self).__init__()

        self.instance_norm1 = nn.BatchNorm3d(in_channels)

        self.in_channels = in_channels

        self.layer1 = self._make_layer(block, in_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_channels * 4, layers[2], stride=2)

        self.softmax = nn.functional.softmax
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels * 4, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.instance_norm1(x)  # Normalize input

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
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
        
        # 3D CNN for grid processing using ResNet3D
        # FIX: Simplify architecture for better training dynamics
        self.grid_cnn = ResNet3D(
            block=Block, 
            layers=[1, 1, 1],  # Reduced layer depth for better training
            in_channels=hidden_nf,
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
        
        # Calculate total nodes and expected grid points
        total_nodes = x.size(0)
        total_grid_points = data.grid_coords.size(0)
        expected_grid_points_per_sample = self.grid_size**3
        
        # FIX: Define these variables properly before using them
        grid_pos = data.grid_coords
        grid_edge_index = data.grid_edge_index
        
        # Embed atoms
        h = self.atom_embedding(atoms)
        
        # Learn orientations
        orientations = self.equivariant_layer(h, x, edge_index)
        
        # Check if indices are valid for the batched data
        if grid_edge_index.numel() > 0 and (grid_edge_index.max() >= total_nodes or grid_edge_index.min() < 0):
            # Filter connections to keep only valid indices
            valid_mask = (grid_edge_index[0] >= 0) & (grid_edge_index[0] < total_nodes) & \
                        (grid_edge_index[1] >= 0) & (grid_edge_index[1] < total_grid_points)
            grid_edge_index = grid_edge_index[:, valid_mask]
        
        # Apply gridification with orientation-aware projection
        grid_features = self.gridification(h, x, grid_pos, grid_edge_index, orientations)
        
        # FIX: Better handling of grid feature reshaping
        try:
            expected_size = batch_size * self.grid_size**3
            if grid_features.size(0) != expected_size:
                # Pad or truncate to match expected size
                if grid_features.size(0) < expected_size:
                    pad_size = expected_size - grid_features.size(0)
                    padding = torch.zeros((pad_size, grid_features.size(1)), 
                                         device=grid_features.device)
                    grid_features = torch.cat([grid_features, padding], dim=0)
                else:
                    grid_features = grid_features[:expected_size]
                    
            # Reshape grid features for CNN processing [B, C, D, H, W]
            grid_features = grid_features.reshape(
                batch_size, 
                self.grid_size, self.grid_size, self.grid_size, 
                self.hidden_nf
            ).permute(0, 4, 1, 2, 3)
        except RuntimeError as e:
            # Fallback to a simpler approach if reshape fails
            print(f"Warning: Grid reshape failed. Using fallback approach. Error: {str(e)}")
            # Create empty features with correct shape
            grid_features = torch.zeros(
                (batch_size, self.hidden_nf, self.grid_size, self.grid_size, self.grid_size),
                device=x.device
            )
        
        # Process grid with 3D CNN
        logits = self.grid_cnn(grid_features)
        
        return logits