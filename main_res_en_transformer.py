import argparse
import os
from torch_scatter import scatter_mean
import torch
from torch import nn

from models.en_transformer.en_transformer import EnTransformer
from residue_common import setup_training, train_model

class EnTransformerResidueClassifier(nn.Module):
    def __init__(self, dim, depth, dim_head=64, heads=8, checkpoint=False):
        super().__init__()
        self.input_embedding = nn.Linear(5, dim)
        self.transformer = EnTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            neighbors=0,  # We'll handle neighbors through edge_index
            only_sparse_neighbors=True,
            checkpoint=checkpoint
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 20)
        )
    
    def forward(self, feat, coors, edge_index):
        feat = self.input_embedding(feat)
        # Create adjacency matrix from edge_index
        adj_mat = torch.zeros(coors.size(0), coors.size(0), device=coors.device)
        adj_mat[edge_index[:, 0], edge_index[:, 1]] = 1
        
        feat_out, _ = self.transformer(feat, coors, adj_mat=adj_mat)
        return self.mlp(feat_out)

def main(args):
    # Only pass the parameters that the classifier expects
    model_kwargs = {
        'dim': args.dim,
        'depth': args.depth,
        'dim_head': args.dim_head,
        'heads': args.heads,
        'checkpoint': args.checkpoint
    }
    
    model, optimizer, criterion, train_loader, val_loader, test_loader = \
        setup_training(args, EnTransformerResidueClassifier, model_kwargs)
    
    train_model(args, model, optimizer, criterion, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--neighbors', type=int, default=32)
    parser.add_argument('--checkpoint', action='store_true')
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