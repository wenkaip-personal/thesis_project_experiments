import argparse
import os
from torch_scatter import scatter_mean
import torch
from torch import nn

from models.en_transformer.en_transformer import EnTransformer
from residue_common import setup_training, train_model

class EnTransformerResidueClassifier(nn.Module):
    def __init__(self, dim, depth, num_tokens=None, dim_head=64, heads=8, neighbors=0, checkpoint=False):
        super().__init__()
        self.input_embedding = nn.Linear(5, dim)
        self.transformer = EnTransformer(
            dim=dim, depth=depth, num_tokens=num_tokens,
            dim_head=dim_head, heads=heads, neighbors=neighbors,
            checkpoint=checkpoint
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 20)
        )
        
    def forward(self, feat, coors, mask=None):
        feat = self.input_embedding(feat)
        feat_out, _ = self.transformer(feat, coors, mask=mask)
        feat_pool = scatter_mean(feat_out, batch=torch.arange(coors.size(0), 
                               device=coors.device).repeat_interleave(coors.size(1)), dim=0)
        return self.mlp(feat_pool)

def main(args):
    model_kwargs = {
        'dim': args.dim,
        'depth': args.depth,
        'dim_head': args.dim_head,
        'heads': args.heads,
        'neighbors': args.neighbors,
        'checkpoint': args.checkpoint
    }
    
    model, optimizer, criterion, train_loader, val_loader, test_loader = \
        setup_training(args, EnTransformerResidueClassifier, model_kwargs)
    
    train_model(args, model, optimizer, criterion, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1')
    parser.add_argument('--batch_size', type=int, default=8)
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
    args = parser.parse_args()

    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(os.path.join(args.outf, args.exp_name), exist_ok=True)

    main(args)