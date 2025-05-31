import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph, knn, radius
from torch_scatter import scatter_mean, scatter_add

class LocalEquivariantLayer(nn.Module):
    """Learn orientations at residue level instead of batch level"""
    def __init__(self, hidden_features):
        super(LocalEquivariantLayer, self).__init__()
        self.hidden_features = hidden_features
        
        # Networks for learning local orientations
        self.orient_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 6)  # Two 3D vectors
        )
        
    def forward(self, x, pos, batch, ca_idx):
        # Learn orientation vectors for each CA atom
        ca_features = x[ca_idx]
        orient_features = self.orient_net(ca_features)
        
        # Split into two vectors
        vec1 = orient_features[:, :3]
        vec2 = orient_features[:, 3:6]
        
        # Apply Gram-Schmidt orthogonalization
        orientations = self.gram_schmidt_batch(vec1, vec2)
        
        return orientations
    
    def gram_schmidt_batch(self, v1, v2):
        # Normalize first vector
        n1 = F.normalize(v1, dim=-1)
        
        # Make second vector orthogonal to first
        v2_proj = (n1 * v2).sum(dim=-1, keepdim=True) * n1
        v2_orth = v2 - v2_proj
        n2 = F.normalize(v2_orth, dim=-1)
        
        # Third vector is cross product
        n3 = torch.cross(n1, n2, dim=-1)
        
        # Stack to form rotation matrices
        return torch.stack([n1, n2, n3], dim=-2)

class GridificationLayer(nn.Module):
    """Gridification layer with proper edge construction"""
    def __init__(self, hidden_features=128, k_atoms_to_grid=3, k_grid_to_atoms=3, radius=4.5):
        super().__init__()
        self.hidden_features = hidden_features
        self.k_atoms_to_grid = k_atoms_to_grid
        self.k_grid_to_atoms = k_grid_to_atoms
        self.radius = radius
        
        # Message passing networks
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_features + 1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        
    def forward(self, atom_features, atom_pos, grid_pos, atom_batch, grid_batch):
        # Build bipartite edges
        edge_index = self.build_bipartite_edges(atom_pos, grid_pos, atom_batch, grid_batch)
        
        # Compute messages
        row, col = edge_index
        distances = torch.norm(atom_pos[row] - grid_pos[col], dim=-1, keepdim=True)
        
        # Message features
        msg_input = torch.cat([
            atom_features[row],
            torch.zeros(col.size(0), self.hidden_features, device=atom_features.device),
            distances
        ], dim=-1)
        
        messages = self.message_net(msg_input)
        
        # Aggregate messages to grid points
        grid_features = scatter_mean(messages, col, dim=0, dim_size=grid_pos.size(0))
        grid_features = self.update_net(grid_features)
        
        return grid_features
    
    def build_bipartite_edges(self, atom_pos, grid_pos, atom_batch, grid_batch):
        # KNN from atoms to grid
        row1, col1 = knn(grid_pos, atom_pos, self.k_atoms_to_grid, 
                         batch_x=grid_batch, batch_y=atom_batch)
        
        # KNN from grid to atoms
        row2, col2 = knn(atom_pos, grid_pos, self.k_grid_to_atoms,
                         batch_x=atom_batch, batch_y=grid_batch)
        
        # Radius edges
        row3, col3 = radius(atom_pos, grid_pos, self.radius,
                           batch_x=atom_batch, batch_y=grid_batch)
        
        # Combine all edges
        row = torch.cat([row1, col2, row3])
        col = torch.cat([col1, row2, col3])
        
        edge_index = torch.stack([row, col])
        return torch_geometric.utils.coalesce(edge_index)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN3D(nn.Module):
    def __init__(self, in_channels, num_classes=20):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=1)
        self.layer3 = self._make_layer(256, 512, 2, stride=1)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class ProteinGrid(nn.Module):
    def __init__(self, node_types=9, res_types=21, on_bb=2, hidden_features=128, out_features=20, act=nn.SiLU):
        super().__init__()
        
        # Embeddings
        self.atom_embedding = nn.Embedding(node_types, 16)
        self.res_embedding = nn.Embedding(res_types, 32)
        self.on_bb_embedding = nn.Embedding(on_bb, 8)
        
        # Feature projection
        input_dim = 16 + 32 + 8 + 3 + 2  # embeddings + coords + physical features
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_features),
            act(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, hidden_features),
        )
        
        # Equivariant orientation learning
        self.equi_layer = LocalEquivariantLayer(hidden_features)
        
        # Gridification
        self.gridification = GridificationLayer(
            hidden_features=hidden_features,
            k_atoms_to_grid=5,
            k_grid_to_atoms=5,
            radius=4.5
        )
        
        # 3D CNN for grid processing
        self.cnn_model = CNN3D(hidden_features, num_classes=out_features)
        
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, batch):
        # Extract features
        atom_pos = batch.coords
        grid_pos = batch.grid_coords
        atom_batch = batch.batch
        
        # Create grid batch indices
        batch_size = batch.batch.max().item() + 1
        grid_points_per_sample = batch.num_grid_points[0].item()
        grid_batch = torch.arange(batch_size, device=atom_pos.device).repeat_interleave(grid_points_per_sample)
        
        # Prepare features
        physical_feats = torch.stack([batch.sasa, batch.charges], dim=-1)
        physical_feats = torch.nan_to_num(physical_feats, 0.0)
        
        # Embeddings
        atom_emb = self.atom_embedding(batch.atom_types)
        res_emb = self.res_embedding(batch.res_types)
        bb_emb = self.on_bb_embedding(batch.atom_on_bb)
        
        # Combine features
        atom_features = self.feature_embedding(
            torch.cat([atom_emb, res_emb, bb_emb, atom_pos, physical_feats], dim=-1)
        )
        
        # Learn orientations at CA positions
        ca_indices = batch.ca_idx
        orientations = self.equi_layer(atom_features, atom_pos, atom_batch, ca_indices)
        
        # Transform grid coordinates using orientations
        grid_size = batch.grid_size[0].item()
        transformed_grids = []
        
        for i in range(batch_size):
            # Get original grid for this sample
            start_idx = i * grid_points_per_sample
            end_idx = (i + 1) * grid_points_per_sample
            sample_grid = grid_pos[start_idx:end_idx]
            
            # Transform using the orientation for this sample
            R = orientations[i]
            transformed_grid = torch.matmul(sample_grid, R.T)
            transformed_grids.append(transformed_grid)
        
        transformed_grid_pos = torch.cat(transformed_grids, dim=0)
        
        # Gridification
        grid_features = self.gridification(
            atom_features, atom_pos, transformed_grid_pos, 
            atom_batch, grid_batch
        )
        
        # Reshape for CNN
        grid_features = grid_features.reshape(
            batch_size, grid_size, grid_size, grid_size, self.hidden_features
        ).permute(0, 4, 1, 2, 3)
        
        # CNN processing
        logits = self.cnn_model(grid_features)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute loss and metrics
        loss = F.cross_entropy(logits, batch.y, reduction='none')
        pred_labels = logits.argmax(dim=-1)
        acc = (pred_labels == batch.y).float()
        
        # Per-class accuracy
        log_dict = {
            "loss": loss,
            "acc": acc
        }
        
        # Track per-class performance
        for i in range(self.out_features):
            mask = batch.y == i
            if mask.any():
                class_acc = (pred_labels[mask] == i).float().mean()
                log_dict[f"acc_{i}"] = class_acc
        
        return loss.mean(), log_dict