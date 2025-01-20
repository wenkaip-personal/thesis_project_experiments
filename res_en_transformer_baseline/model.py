import torch
import torch.nn as nn
from en_transformer import EnTransformer

class RESEnTransformer(nn.Module):
    def __init__(
        self,
        num_atom_types=5,     # C, N, O, S, P
        hidden_dim=128,
        n_layers=4,
        heads=8,
        dim_head=32,
        num_classes=20,       # 20 amino acid classes
        dropout=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(RESEnTransformer, self).__init__()

        # Initialize En Transformer backbone
        self.transformer = EnTransformer(
            dim=hidden_dim,
            depth=n_layers,
            dim_head=dim_head,
            heads=heads,
            num_tokens=num_atom_types,
            rel_pos_emb=True,
            dropout=dropout
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        self.device = device
        self.to(device)
    
    def forward(self, atom_feats, coords, mask=None):
        # Move inputs to device
        atom_feats = atom_feats.to(self.device) 
        coords = coords.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Process through En Transformer
        # Note: atoms are treated as tokens, coordinates are used for spatial attention
        feats, _ = self.transformer(atom_feats, coords, mask=mask)
        
        # Get prediction for central residue (first atom)
        central_feats = feats[:, 0]
        
        # Get class predictions 
        logits = self.pred_head(central_feats)
        
        return logits