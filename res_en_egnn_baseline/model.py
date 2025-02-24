import os
import sys
import torch
import torch.nn as nn

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.egnn.egnn_clean import EGNN

class ResEGNN(nn.Module):
    def __init__(self, in_node_nf=9, hidden_nf=128, out_node_nf=20, in_edge_nf=16, n_layers=4, device='cuda'):
        """
        Adapts EGNN for the RES task, following ATOM3D implementation pattern.
        
        Args:
            in_node_nf: Input node feature dimension (9 for one-hot encoded atoms)
            hidden_nf: Hidden dimension size
            out_node_nf: Output dimension (20 for amino acid classification)
            in_edge_nf: Input edge feature dimension (16 for RBF features)
            n_layers: Number of EGNN layers
            device: Device to run model on
        """
        super().__init__()
        
        # Core EGNN network
        self.egnn = EGNN(in_node_nf=in_node_nf,
                        hidden_nf=hidden_nf,
                        out_node_nf=hidden_nf,
                        in_edge_nf=in_edge_nf,
                        n_layers=n_layers,
                        device=device)

        # Final MLP matching ATOM3D implementation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, 2*hidden_nf),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.1),
            nn.Linear(2*hidden_nf, out_node_nf)
        )
        
        self.to(device)

    def forward(self, h, x, edges, batch):
        """
        Forward pass following ATOM3D pattern for RES task.
        
        Args:
            h: Node features [n_nodes, in_node_nf]
            x: Node coordinates [n_nodes, 3]
            edges: Graph connectivity [2, n_edges]
            batch: Graph batch containing ca_idx for central residue position
        """
        # Get edge attributes from batch
        edge_attr = batch.edge_s
            
        # Get node embeddings from EGNN
        h, x = self.egnn(h, x, edges, edge_attr)

        # Get class logits
        out = self.mlp(h)

        # Extract central alpha carbon indices
        central_residue_idx = batch.ca_idx + batch.ptr[:-1]
        
        return out[central_residue_idx]