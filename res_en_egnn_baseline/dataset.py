import torch
from torch.utils.data import Dataset
import atom3d.datasets as da
import atom3d.util.graph as gr
import atom3d.util.transforms as tr

class RESDataset(Dataset):
    """Dataset for residue identity prediction using EGNN."""
    
    def __init__(self, lmdb_path, transform=None):
        self.dataset = da.load_dataset(lmdb_path, 'lmdb')
        self.transform = transform if transform else tr.GraphTransform(atom_key='atoms', label_key='labels')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        graph = self.transform(item)
        
        # Extract necessary components for EGNN
        x = graph.pos  # Node coordinates
        h = graph.node_feats  # Node features  
        edge_index = graph.edge_index  # Graph connectivity
        y = graph.y  # Residue labels
        
        return h, x, edge_index, y