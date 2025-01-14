import os
import sys
import torch
import torch.nn as nn

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.egnn.egnn_clean import EGNN

class RESEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf=18,  # Change to match actual input features
        hidden_nf=128,  # Hidden dimension
        out_node_nf=20,  # Number of residue classes
        n_layers=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(RESEGNN, self).__init__()
        
        # Initialize EGNN backbone
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            n_layers=n_layers,
            device=device
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, out_node_nf)
        )
        
        self.device = device
        self.to(device)
    
    def forward(self, node_feats, pos, edge_index, edge_attrs=None):
        # Ensure edge_attrs has correct dimensions if not None
        if edge_attrs is not None and edge_attrs.dim() == 1:
            edge_attrs = edge_attrs.unsqueeze(-1)

        # Process through EGNN
        node_feats, _ = self.egnn(node_feats, pos, edge_index, edge_attrs)
        
        # Get prediction for central residue
        central_feats = node_feats[0]  # The first node is the central residue
        
        # Get class predictions
        logits = self.pred_head(central_feats)
        
        return logits