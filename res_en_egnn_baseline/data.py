import torch
from torch.utils.data import Dataset, DataLoader
import atom3d.datasets as da
import atom3d.util.transforms as tr
import atom3d.util.graph as gr

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
        
        # Convert to PyTorch tensors
        node_feats = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'node_feats': node_feats,
            'edge_index': edge_index,  
            'edge_attrs': edge_attrs,
            'pos': pos,
            'label': label
        }

def get_res_dataloaders(train_path, val_path, test_path, batch_size=32, num_workers=4):
    """
    Creates dataloaders for RES dataset splits
    """
    # Create datasets
    train_dataset = RESDataset(train_path)
    val_dataset = RESDataset(val_path) 
    test_dataset = RESDataset(test_path)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader