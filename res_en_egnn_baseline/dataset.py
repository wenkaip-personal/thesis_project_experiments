import torch
from torch.utils.data import Dataset
import atom3d.datasets as da
import atom3d.util.graph as gr
import atom3d.util.transforms as tr
from torch_geometric.data import Data, Batch

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
        
        # Convert the label to a tensor if it isn't already
        if isinstance(graph.y, list):
            label = torch.tensor(graph.y, dtype=torch.long)
        else:
            label = graph.y
        
        # Create a PyTorch Geometric Data object
        data = Data(
            x=graph.x,  # Node features
            pos=graph.pos,  # Node coordinates 
            edge_index=graph.edge_index,  # Graph connectivity
            edge_attr=graph.edge_attr,  # Edge attributes/weights
            y=label  # Residue labels as tensor
        )
        
        return data