import torch
from torch.utils.data import Dataset
import atom3d.datasets as da
import atom3d.util.graph as gr
import atom3d.util.transforms as tr
from torch_geometric.data import Data

class RESDataset(Dataset):
    """Dataset for residue identity prediction using EGNN."""
    
    def __init__(self, lmdb_path, transform=None):
        self.dataset = da.load_dataset(lmdb_path, 'lmdb')
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # First get the central residue indices from the raw dataframe
        central_indices = []
        for indices in item['subunit_indices']:
            # Find the CA atom index for each central residue
            ca_idx = None
            for idx in indices:
                if item['atoms'].iloc[idx]['name'] == 'CA':
                    ca_idx = idx
                    break
            if ca_idx is not None:
                central_indices.append(ca_idx)
        
        # Then apply the graph transform
        if self.transform is None:
            self.transform = tr.GraphTransform(atom_key='atoms', label_key='labels')
        graph = self.transform(item)
        
        label = torch.tensor(graph.y['label'], dtype=torch.long)
        
        # Create a PyTorch Geometric Data object
        data = Data(
            x=graph.x,  # Node features
            pos=graph.pos,  # Node coordinates 
            edge_index=graph.edge_index,  # Graph connectivity
            edge_attr=graph.edge_attr,  # Edge attributes/weights
            y=label,  # Residue labels
            central_indices=torch.tensor(central_indices, dtype=torch.long)  # Indices of central residues
        )

        return data