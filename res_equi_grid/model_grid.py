import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool, radius_graph, knn, radius

class equivariant_layer(nn.Module):
    def __init__(self, hidden_features):
        super(equivariant_layer, self).__init__()
        self.hidden_features = hidden_features
        self.message_net1 = nn.Sequential(
            nn.Linear(2*hidden_features+1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.message_net2 = nn.Sequential(
            nn.Linear(2*hidden_features+1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.p = 5

    def forward(self, x, pos, batch):
        edge_index = radius_graph(pos, r=4.5, batch=batch, loop=True)
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1, keepdim=True)
        vec1, vec2 = self.message(x[edge_index[0]], x[edge_index[1]], dist, pos[edge_index[0]], pos[edge_index[1]])
        vec1_out, vec2_out = global_add_pool(vec1, edge_index[0]), global_add_pool(vec2, edge_index[0])
        vec1_out = global_mean_pool(vec1_out, batch)
        vec2_out = global_mean_pool(vec2_out, batch)
        return self.gram_schmidt_batch(vec1_out, vec2_out)

    def gram_schmidt_batch(self, v1, v2):
        n1 = v1 / (torch.norm(v1, dim=-1, keepdim=True)+1e-8)
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2 = n2_prime / (torch.norm(n2_prime, dim=-1, keepdim=True)+1e-8)
        n3 = torch.cross(n1, n2, dim=-1)
        return torch.stack([n1, n2, n3], dim=-2)
    
    def omega(self, dist):
        out = 1 - (self.p+1)*(self.p+2)/2 * (dist/4.5)**self.p + self.p*(self.p+2) * (dist/4.5)**(self.p+1) - self.p*(self.p+1)/2 * (dist/4.5)**(self.p+2)
        return out
    
    def message(self, x_i, x_j, dist, pos_i, pos_j):
        x_ij = torch.cat([x_i, x_j, dist], dim=-1)
        mes_1 = self.message_net1(x_ij)
        mes_2 = self.message_net2(x_ij)
        coe = self.omega(dist)
        norm_vec = (pos_i - pos_j) / (torch.norm(pos_i - pos_j, dim=-1, keepdim=True)+1e-8)
        return norm_vec * coe * mes_1, norm_vec * coe * mes_2

class MPNNLayer(nn.Module):
    """ Message Passing Layer """
    def __init__(self, edge_features=6, hidden_features=128, act=nn.SiLU):
        super().__init__()
        self.edge_model = nn.Sequential(nn.Linear(edge_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, hidden_features))
        
        self.message_model = nn.Sequential(nn.Linear(hidden_features*2, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))

        self.update_net =  nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))
        
    def forward(self, node_embedding, node_pos, grid_pos, edge_index):
        message = self.message(node_embedding, node_pos, grid_pos, edge_index)
        x = self.update(message, edge_index[1])
        return x

    def message(self, node_embedding, node_pos, grid_pos, edge_index):
        index_i, index_j = edge_index[0], edge_index[1]
        pos_nodes, pos_grids = node_pos[index_i], grid_pos[index_j]
        edge_attr = torch.cat((pos_nodes, pos_grids), dim=-1)
        pos_embedding = self.edge_model(edge_attr)
        node_embedding = node_embedding[index_i]
        message = torch.cat((node_embedding, pos_embedding), dim=-1)
        message = self.message_model(message)
        return message

    def update(self, message, index_j):
        """ Update node features """
        num_messages = torch.bincount(index_j)
        message = global_add_pool(message, index_j) / num_messages.unsqueeze(-1)
        update = self.update_net(message)

        return update

class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 20, dropout: float = 0.5):
        super(SimpleCNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool3d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ProteinGrid(nn.Module):
    def __init__(self, node_types=4, res_types=21, on_bb=2, hidden_features=128, out_features=20, act=nn.SiLU):
        super().__init__()
        self.atom_embedding = nn.Embedding(node_types, node_types)
        self.res_embedding = nn.Embedding(res_types, res_types)
        self.on_bb_embedding = nn.Embedding(on_bb, on_bb)
        self.feature_embedding = nn.Sequential(
            nn.Linear(node_types + res_types + on_bb + 3 + 2, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )
        
        self.equi_layer = equivariant_layer(hidden_features)
        self.mpnn_layer = MPNNLayer(hidden_features=hidden_features, act=act)

        self.cnn_model = SimpleCNN3D(
            in_channels=hidden_features, 
            num_classes=out_features,
            dropout=0.5
        )
        self.hidden_features = hidden_features

    def forward(self, batch):
        batch_size = batch.ptr.shape[0] - 1
        node_pos = batch.coords.to(torch.float32)
        grid_pos = batch.grid_coords.to(torch.float32)
        physical_feats = torch.stack([batch.sasa, batch.charges], dim=-1).to(torch.float32)
        physical_feats[torch.isinf(physical_feats)] = 0
        physical_feats[torch.isnan(physical_feats)] = 0
        atom_types = batch.atom_types.to(torch.long)   
        atom_on_bb = batch.atom_on_bb.to(torch.long)
        res_types = batch.res_types.to(torch.long)
        atom_embedding = self.atom_embedding(atom_types)
        res_embedding = self.res_embedding(res_types)
        on_bb_embedding = self.on_bb_embedding(atom_on_bb)
        atom_feature = self.feature_embedding(
            torch.cat((atom_embedding, res_embedding, on_bb_embedding, node_pos, physical_feats), dim=-1)
        )
        
        atom_batch = batch.batch[batch.is_atom_mask.bool() == True]  # Get batch assignments for atoms only
        frame = self.equi_layer(atom_feature, node_pos, atom_batch)

        # Get the number of grid points per sample
        grid_points_per_sample = batch.grid_size[0]**3
        
        # Create batch indices for grid points
        grid_batch_idx = torch.arange(batch_size, device=grid_pos.device).repeat_interleave(grid_points_per_sample)
        
        # Since grid_pos contains concatenated grid coordinates from all samples,
        # we need to select the appropriate grid coordinates for transformation
        # The first grid_points_per_sample points are the actual grid coordinates
        grid_coords_single = grid_pos[:grid_points_per_sample]
        
        # Repeat grid coordinates for each sample in the batch
        grid_pos_batched = grid_coords_single.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply frame transformation
        grid_pos = torch.bmm(grid_pos_batched, frame.permute(0, 2, 1)).reshape(-1, 3)

        row_1, col_1 = knn(node_pos, grid_pos, k=3, batch_x=atom_batch, batch_y=grid_batch_idx)
        row_2, col_2 = knn(grid_pos, node_pos, k=3, batch_x=grid_batch_idx, batch_y=atom_batch)

        edge_index_knn = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )

        row_1, col_1 = radius(node_pos, grid_pos, r=4, batch_x=atom_batch, batch_y=grid_batch_idx)
        row_2, col_2 = radius(grid_pos, node_pos, r=4, batch_x=grid_batch_idx, batch_y=atom_batch)

        edge_index_radius = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )
        edge_index = torch.cat((edge_index_knn, edge_index_radius), dim=-1)

        edge_index = torch_geometric.utils.coalesce(edge_index)

        cnn_input = self.mpnn_layer(atom_feature, node_pos, grid_pos, edge_index)
        cnn_input = cnn_input.reshape(
            batch_size, 
            int(batch.grid_size[0]), int(batch.grid_size[0]), int(batch.grid_size[0]), 
            self.hidden_features
        ).permute(0, 4, 1, 2, 3)
        preds = self.cnn_model(cnn_input)
        preds = F.log_softmax(preds, dim=-1)
        loss = F.cross_entropy(preds, batch.y, reduction='none')
        pred_labels = torch.max(preds, dim=-1)[1]
        acc = (pred_labels == batch.y).float()
        backprop_loss = loss.mean()  # []

        correct_counts = torch.zeros(20).to(batch.y.device)
        correct_counts.scatter_add_(0, batch.y, (pred_labels == batch.y).float())
        total_counts = torch.zeros(20).to(batch.y.device)
        total_counts.scatter_add_(0, batch.y, torch.ones_like(batch.y).float())
        acc_per_class = correct_counts / total_counts
        log_dict = dict()
        log_dict["loss"] = loss
        log_dict["acc"] = acc
        for i, acc_cls in enumerate(acc_per_class):
            log_dict[f"acc_{i}"] = torch.ones(batch_size).to(batch.y.device)*acc_cls

        return backprop_loss, log_dict