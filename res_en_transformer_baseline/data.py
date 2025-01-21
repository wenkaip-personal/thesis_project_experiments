import torch
from torch.utils.data import Dataset
import atom3d.datasets as da
import atom3d.util.transforms as tr
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class RESTransformerDataset(Dataset):
    def __init__(self, lmdb_path):
        self.dataset = da.load_dataset(lmdb_path, 'lmdb')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract the atoms dataframe and label
        atoms_df = item['atoms'] 
        label = item['labels']['label'].values[0]  # Get the residue label
        
        # Extract positions and atomic features
        coords = atoms_df[['x', 'y', 'z']].values
        atom_types = atoms_df['element'].values
        
        # One-hot encode atom types
        unique_atoms = ['C', 'N', 'O', 'S', 'P']  # Common atoms in proteins
        atom_feats = torch.zeros((len(atom_types), len(unique_atoms)))
        for i, atom in enumerate(atom_types):
            if atom in unique_atoms:
                atom_feats[i, unique_atoms.index(atom)] = 1.0
        
        # Convert to PyTorch tensors
        coords = torch.tensor(coords, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        # Return tensors in format expected by En Transformer
        return {
            'feats': atom_feats,
            'coords': coords,
            'label': label
        }

def get_res_dataloaders(train_path, val_path, test_path, batch_size=32, num_workers=4):
    """
    Creates dataloaders for RES dataset splits
    """
    # Create datasets
    train_dataset = RESTransformerDataset(train_path)
    val_dataset = RESTransformerDataset(val_path)
    test_dataset = RESTransformerDataset(test_path)
    
    # Create dataloaders with custom collate function
    def collate_fn(batch):
        # Get max sequence length in this batch
        max_atoms = max(item['feats'].size(0) for item in batch)
        batch_size = len(batch)
        
        # Get dimensions of features
        feat_dim = batch[0]['feats'].size(1)
        
        # Initialize padded tensors
        padded_feats = torch.zeros(batch_size, max_atoms, feat_dim)
        padded_coords = torch.zeros(batch_size, max_atoms, 3)
        mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
        labels = torch.stack([item['label'] for item in batch])
        
        # Fill padded tensors
        for i, item in enumerate(batch):
            n_atoms = item['feats'].size(0)
            padded_feats[i, :n_atoms] = item['feats']
            padded_coords[i, :n_atoms] = item['coords']
            mask[i, :n_atoms] = True  # True indicates valid atoms
            
        return padded_feats, padded_coords, labels, mask
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader