import torch
from torch.utils.data import Dataset
import atom3d.datasets as da
import atom3d.util.transforms as tr
import atom3d.util.graph as gr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class RESDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.dataset = da.load_dataset(lmdb_path, 'lmdb')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract the atoms dataframe and label
        atoms_df = item['atoms']
        label = item['labels']['label'].values[0]  # Get the residue label
        
        # Convert atoms to graph representation
        node_feats, edge_index, edge_attrs, pos = gr.prot_df_to_graph(atoms_df)

        # Add extra dimension to edge_attrs if it's 1D
        if edge_attrs.dim() == 1:
            edge_attrs = edge_attrs.unsqueeze(-1)
        
        # Return PyG Data object
        return Data(
            x=node_feats,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            pos=pos,
            y=label
        )

def get_res_dataloaders(train_path, val_path, test_path, batch_size=32, num_workers=4, debug=False):
    """
    Creates dataloaders for RES dataset splits using PyG's DataLoader
    """
    # Create datasets
    train_dataset = RESDataset(train_path)
    val_dataset = RESDataset(val_path) 
    test_dataset = RESDataset(test_path)
    
    if debug:
        # Take only first 100 samples for each split when debugging
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(50))
        test_dataset = torch.utils.data.Subset(test_dataset, range(50))
    
    # Create dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader