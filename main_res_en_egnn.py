import argparse
import os
from torch_scatter import scatter_mean
import torch
from torch import nn

from models.egnn.egnn_clean import EGNN, get_edges_batch
from residue_common import setup_training, train_model

class EGNNResidueClassifier(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers, attention):
        super().__init__()
        self.egnn = EGNN(in_node_nf=in_node_nf,
                        hidden_nf=hidden_nf,
                        out_node_nf=hidden_nf,
                        n_layers=n_layers,
                        attention=attention)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, out_node_nf)
        )
    
    def forward(self, h, x, edge_index):
        # Convert edge_index to row, col format
        if edge_index.dim() == 2:
            row, col = edge_index[:, 0], edge_index[:, 1]
        else:
            row, col = edge_index
        
        # Create edge_index in format [2, num_edges]
        edges = torch.stack([row, col], dim=0)
        edge_attr = torch.ones(edges.size(1), 1).to(edges.device)
        
        h_out, x_out = self.egnn(h, x, edges, edge_attr)
        return self.mlp(h_out)

def main(args):
    model_kwargs = {
        'in_node_nf': 5,
        'hidden_nf': args.nf,
        'out_node_nf': 20,
        'n_layers': args.n_layers,
        'attention': args.attention
    }
    
    model, optimizer, criterion, train_loader, val_loader, test_loader = \
        setup_training(args, EGNNResidueClassifier, model_kwargs)
    
    train_model(args, model, optimizer, criterion, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--attention', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--outf', type=str, default='res_outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--radius', type=float, default=10.0)
    parser.add_argument('--use_knn', type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(os.path.join(args.outf, args.exp_name), exist_ok=True)

    main(args)