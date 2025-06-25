import os
import sys
import torch
import torch.nn as nn

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.non_equi_transformer.non_equi_transformer import EnTransformer

class ResEnTransformer(nn.Module):
    def __init__(self, input_nf = 9, output_nf = 20, hidden_nf = 128, n_layers=4, n_heads=4, device='cuda'):
        super().__init__()
        
        # Add embedding layer to convert atom indices to feature vectors
        self.embed = nn.Embedding(input_nf, input_nf)
        
        # Main En Transformer network - now expecting hidden_nf as input dimension
        self.transformer = EnTransformer(
            input_nf=input_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        # Final MLP matching ATOM3D implementation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, 2*hidden_nf),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.1),
            nn.Linear(2*hidden_nf, output_nf)
        )
        
        self.to(device)

    def forward(self, h, x, edges, batch):
        # Embed atom types to convert from Long to Float tensor with proper dimensions
        h = self.embed(h)
        
        # Create an edge mask: True if nodes are in the same graph, False otherwise
        # This ensures attention only happens within each protein graph
        edge_src, edge_dst = edges
        edge_mask = batch.batch[edge_src] == batch.batch[edge_dst]
        
        # Apply En Transformer with edge mask
        h, x = self.transformer(h, x, edges, mask=edge_mask)

        # Get class logits
        out = self.mlp(h)
        
        # Extract embeddings for the alpha carbon atoms
        return out[batch.ca_idx + batch.ptr[:-1]]