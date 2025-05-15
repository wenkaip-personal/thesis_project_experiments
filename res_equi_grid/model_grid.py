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
        # Use a larger epsilon for better numerical stability
        eps = 1e-6
        
        # Normalize first vector
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)
        mask1 = (v1_norm > eps).float()
        n1 = v1 / (v1_norm + eps)
        
        # Handle zero norm vectors by using a random unit vector
        if torch.any(v1_norm < eps):
            random_vectors = torch.randn_like(v1)
            random_vectors = random_vectors / torch.norm(random_vectors, dim=-1, keepdim=True)
            n1 = torch.where(mask1 > 0.5, n1, random_vectors)
        
        # Make second vector orthogonal to first
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2_norm = torch.norm(n2_prime, dim=-1, keepdim=True)
        mask2 = (n2_norm > eps).float()
        
        # Default orthogonal vector for numerical stability
        # Create a vector orthogonal to n1 using a simple rotation
        default_orth = torch.zeros_like(n1)
        default_orth[:, 0] = -n1[:, 1]
        default_orth[:, 1] = n1[:, 0]
        default_orth[:, 2] = 0
        
        # normalize the default orthogonal vector
        default_norm = torch.norm(default_orth, dim=-1, keepdim=True)
        default_mask = (default_norm > eps).float()
        default_orth = torch.where(default_mask > 0.5, 
                                  default_orth / (default_norm + eps),
                                  torch.tensor([0, 0, 1], device=n1.device).expand_as(n1))
        
        # Use default if n2_prime is close to zero
        n2 = torch.where(mask2 > 0.5, 
                         n2_prime / (n2_norm + eps),
                         default_orth)
        
        # Create third vector using cross product for right-handed system
        n3 = torch.cross(n1, n2, dim=-1)
        n3_norm = torch.norm(n3, dim=-1, keepdim=True)
        n3 = n3 / (n3_norm + eps)
        
        # Stack to create orientation matrices [N, 3, 3]
        return torch.stack([n1, n2, n3], dim=-1)
    
    def omega(self, dist):
        """Smoothing function for distance weighting"""
        r_max = 4.5  # Maximum radius
        # Using a smooth polynomial cutoff function
        dist_ratio = torch.clamp(dist/r_max, 0, 1)
        out = 1 - (self.p+1)*(self.p+2)/2 * dist_ratio**self.p + \
              self.p*(self.p+2) * dist_ratio**(self.p+1) - \
              self.p*(self.p+1)/2 * dist_ratio**(self.p+2)
        # Ensure output is non-negative
        return torch.clamp(out, 0, 1)
    
    def message(self, x_i, x_j, dist, pos_i, pos_j):
        """Create messages between nodes with numerical safeguards"""
        # Combine features and distance
        x_ij = torch.cat([x_i, x_j, dist], dim=-1)
        
        # Get message weights
        mes_1 = self.message_net1(x_ij)
        mes_2 = self.message_net2(x_ij)
        
        # Apply distance weighting
        coe = self.omega(dist)
        
        # Get normalized direction vector with numerical stability
        direction = pos_i - pos_j
        norm = torch.norm(direction, dim=-1, keepdim=True)
        safe_norm = norm + 1e-8
        norm_vec = direction / safe_norm
        
        # Apply weight to direction
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
        # Safety check: if edge_index is empty, return zero grid features
        if edge_index.numel() == 0:
            return torch.zeros(grid_pos.size(0), node_features.size(1), device=node_pos.device)
            
        # Validate edge_index
        max_node_index = node_pos.size(0) - 1
        max_grid_index = grid_pos.size(0) - 1
        
        valid_source = (edge_index[0] >= 0) & (edge_index[0] <= max_node_index)
        valid_target = (edge_index[1] >= 0) & (edge_index[1] <= max_grid_index)
        
        valid_mask = valid_source & valid_target
        if not torch.all(valid_mask):
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
        
        # Apply orientations to decouple from global rotation
        # [E, 3] @ [E, 3, 3] -> [E, 3]
        # For each edge, we transform the relative position using the orientation of the source point
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
        """Update grid features based on messages with better handling for empty data"""
        # Handle empty case
        if len(target_indices) == 0:
            return torch.zeros(num_grid_points, messages.size(1), device=messages.device)
        
        # Check for invalid indices and filter them out
        valid_mask = (target_indices >= 0) & (target_indices < num_grid_points)
        
        if not torch.all(valid_mask):
            messages = messages[valid_mask]
            target_indices = target_indices[valid_mask]
            
        # If still empty after filtering, return zeros
        if len(target_indices) == 0:
            return torch.zeros(num_grid_points, messages.size(1), device=messages.device)
        
        # Count messages per grid point for normalization
        # Convert to long type for bincount
        target_indices_long = target_indices.long()
        num_messages = torch.bincount(target_indices_long, minlength=num_grid_points).to(messages.device)
        
        # Handle edge case of empty bins
        num_messages = torch.clamp(num_messages, min=1).unsqueeze(-1)
        
        # Aggregate messages for each grid point
        grid_features = scatter_add(messages, target_indices, dim=0, dim_size=num_grid_points)
        
        # Normalize by number of messages
        grid_features = grid_features / num_messages
        
        # Apply update network
        grid_features = self.update_model(grid_features)
        
        return grid_features

class SimpleCNN3D(nn.Module):
    """
    Simplified 3D CNN for processing grid representations
    """
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN3D, self).__init__()
        
        # Simple conv architecture
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class EquivariantGridModel(nn.Module):
    """
    Full model combining equivariant learning, gridification, and 3D CNN
    for residue identity prediction with improved numerical stability.
    """
    def __init__(self, 
                 in_node_nf=9,  # Number of atom types
                 hidden_nf=128, 
                 out_node_nf=20,  # Number of amino acid types
                 grid_size=6,
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
        
        # 3D CNN for grid processing - using simplified CNN
        self.grid_cnn = SimpleCNN3D(
            in_channels=hidden_nf,
            num_classes=out_node_nf
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, atoms, x, edge_index, data):
        """
        Forward pass with better error handling
        
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
        
        # Get grid positions and connection indices
        grid_pos = data.grid_coords
        grid_edge_index = data.grid_edge_index
        
        # Validate edge indices are within bounds
        # Maximum valid indices
        max_node_idx = x.size(0) - 1
        max_grid_idx = grid_pos.size(0) - 1
        
        # Check if any indices are outside valid range
        if grid_edge_index.numel() > 0:
            invalid_src = (grid_edge_index[0] < 0) | (grid_edge_index[0] > max_node_idx)
            invalid_tgt = (grid_edge_index[1] < 0) | (grid_edge_index[1] > max_grid_idx)
            
            if torch.any(invalid_src | invalid_tgt):
                valid_mask = ~(invalid_src | invalid_tgt)
                grid_edge_index = grid_edge_index[:, valid_mask]
        
        # Apply gridification with orientation-aware projection
        grid_features = self.gridification(h, x, grid_pos, grid_edge_index, orientations)
        
        # Count grid points per batch item for correct reshaping
        grid_per_batch = data.num_grid_points
        expected_grid_points = self.grid_size**3
        
        # Reshape grid features for CNN processing [B, C, D, H, W]
        try:
            # Reshape for batched 3D CNN processing
            grid_features_reshaped = []
            
            offset = 0
            for b in range(batch_size):
                # Create consistent size grid array
                batch_grid = torch.zeros(
                    expected_grid_points, 
                    self.hidden_nf, 
                    device=grid_features.device
                )
                
                # Get number of grid points in this batch item
                if hasattr(data, 'num_grid_points'):
                    num_pts = expected_grid_points
                else:
                    # Fallback calculation
                    num_pts = expected_grid_points
                
                # Copy available features
                if offset + num_pts <= grid_features.size(0):
                    batch_grid[:num_pts] = grid_features[offset:offset+num_pts]
                else:
                    # Handle case where we have fewer points than expected
                    available = min(grid_features.size(0) - offset, num_pts)
                    if available > 0:
                        batch_grid[:available] = grid_features[offset:offset+available]
                
                # Reshape to 3D grid
                batch_grid = batch_grid.reshape(
                    self.grid_size, self.grid_size, self.grid_size, self.hidden_nf
                ).permute(3, 0, 1, 2)
                
                grid_features_reshaped.append(batch_grid)
                
                # Update offset for next batch item
                offset += num_pts
                
            # Stack along batch dimension
            grid_features = torch.stack(grid_features_reshaped, dim=0)
            
        except Exception as e:
            # Fallback: create a zero tensor with expected shape
            grid_features = torch.zeros(
                batch_size, 
                self.hidden_nf,
                self.grid_size, self.grid_size, self.grid_size, 
                device=x.device
            )
            
            print(f"Warning: Error during reshape, using zeros. Error: {str(e)}")
        
        # Process grid with 3D CNN
        logits = self.grid_cnn(grid_features)
        
        return logits