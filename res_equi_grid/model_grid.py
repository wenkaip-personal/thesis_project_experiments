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

class EquivariantMPNNLayer(nn.Module):
    """ Equivariant Message Passing Layer """
    def __init__(self, edge_features=6, hidden_features=128, act=nn.SiLU):
        super().__init__()
        self.edge_model = nn.Sequential(nn.Linear(3, hidden_features),  # Just the relative position in local frame
                                        act(),
                                        nn.Linear(hidden_features, hidden_features))
        
        self.message_model = nn.Sequential(nn.Linear(hidden_features*2, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))

        self.update_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, hidden_features))
        
    def forward(self, node_embedding, node_pos, grid_pos, edge_index, equi_frames, batch):
        message = self.message(node_embedding, node_pos, grid_pos, edge_index, equi_frames, batch)
        x = self.update(message, edge_index[1])
        return x

    def message(self, node_embedding, node_pos, grid_pos, edge_index, equi_frames, batch):
        index_i, index_j = edge_index[0], edge_index[1]
        
        # Get source atom positions and target grid positions
        pos_nodes, pos_grids = node_pos[index_i], grid_pos[index_j - node_pos.shape[0]]
        
        # Get equivariant frames for source atoms
        frames = equi_frames[batch[index_i]]
        
        # Compute relative positions in local frame (this is the key equivariant step)
        rel_pos = pos_grids - pos_nodes
        
        # Project relative positions to local frames
        # frames has shape [batch_size, 3, 3] (3 basis vectors)
        # rel_pos has shape [num_edges, 3]
        # Need to batch-wise multiply for correct projection
        local_rel_pos = torch.bmm(frames, rel_pos.unsqueeze(-1)).squeeze(-1)
        
        # Process edge attributes
        pos_embedding = self.edge_model(local_rel_pos)
        node_embedding_i = node_embedding[index_i]
        
        # Combine node and position embeddings
        message = torch.cat((node_embedding_i, pos_embedding), dim=-1)
        message = self.message_model(message)
        return message

    def update(self, message, index_j):
        """ Update node features """
        num_messages = torch.bincount(index_j)
        message = global_add_pool(message, index_j) / num_messages.unsqueeze(-1)
        update = self.update_net(message)
        return update

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm3d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=5, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = nn.ReLU()(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers: list, num_classes: int = 20, in_channels: int = 256):
        super(ResNet3D, self).__init__()

        self.instance_norm1 = nn.BatchNorm3d(in_channels)

        self.in_channels = in_channels

        self.layer1 = self._make_layer(block, in_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, in_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, in_channels * 4, layers[2], stride=1)

        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(in_channels * 4, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.instance_norm1(x)  # (bs, 128, 15, 15, 15)

        x1 = self.layer1(x)  # (bs, 128, 11, 11, 11)
        x2 = self.layer2(x1)  # (bs, 256, 7, 7, 7 )
        x3 = self.layer3(x2) # (bs, 512, 3, 3, 3)
        x_out = F.max_pool3d(x3, kernel_size=x3.shape[-1], stride=3)
        out = x_out.squeeze()
        out = self.fc(out)
        return out

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
        self.mpnn_layer = EquivariantMPNNLayer(hidden_features=hidden_features, act=act)

        self.cnn_model = ResNet3D(
            block=Block, layers=[1, 1, 1, 1], in_channels=hidden_features, num_classes=out_features
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
        edge_index = batch.grid_edge_index
        atom_embedding = self.atom_embedding(atom_types)
        res_embedding = self.res_embedding(res_types)
        on_bb_embedding = self.on_bb_embedding(atom_on_bb)
        atom_feature = self.feature_embedding(
            torch.cat((atom_embedding, res_embedding, on_bb_embedding, node_pos, physical_feats), dim=-1)
        )

        # Compute equivariant frames for each atom
        equi_frames = self.equi_layer(atom_feature, node_pos, batch.batch)
        
        # Use equivariant frames in the MPNN layer
        cnn_input = self.mpnn_layer(atom_feature, node_pos, grid_pos, edge_index, equi_frames, batch.batch)
        
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