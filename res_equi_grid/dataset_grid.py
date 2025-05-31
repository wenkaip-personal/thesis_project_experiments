import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric
from pathlib import Path
import json
from torch_cluster import knn, radius
from typing import Any
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from atom3d.datasets import LMDBDataset
import numpy as np
import random
import math

# Element and amino acid mappings
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)

_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

class GridData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['grid_edge_index', 'edge_index']:
            return self.coords.size(0)
        elif key == 'grid_batch':
            return self.batch.max().item() + 1 if hasattr(self, 'batch') and self.batch.numel() > 0 else 1
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['grid_edge_index', 'edge_index']:
            return 1
        else:
            return 0

class Protein:
    """
    Protein Dataset based on ATOM3D RES dataset
    """

    def __init__(self, lmdb_path, split_path=None, radius=4.5, k=2, knn=True, size=9, spacing=8, max_samples=None):
        self.dataset = LMDBDataset(lmdb_path)
        
        if split_path is not None:
            self.idx = list(map(int, open(split_path).read().split()))
        else:
            self.idx = list(range(len(self.dataset)))
            
        # Build a mapping of valid samples to their subunits
        self.valid_samples = self._build_valid_samples(self.idx)
        
        if max_samples is not None:
            # Limit the total number of samples
            total_samples = sum(len(subunits) for subunits in self.valid_samples.values())
            if total_samples > max_samples:
                # Proportionally reduce samples
                fraction = max_samples / total_samples
                for idx in list(self.valid_samples.keys()):
                    n_subunits = len(self.valid_samples[idx])
                    n_keep = max(1, int(n_subunits * fraction))
                    self.valid_samples[idx] = self.valid_samples[idx][:n_keep]
        
        # Flatten the valid samples for indexing
        self.flat_samples = []
        for idx, subunit_indices in self.valid_samples.items():
            for sub_idx in subunit_indices:
                self.flat_samples.append((idx, sub_idx))
                
        self.knn = knn
        self.radius = radius
        self.k = k
        self.grid_coords = self.generate_grid(n=size, spacing=spacing)
        self.size = size
        self.spacing = spacing
        
        print(f"Loaded {len(self.flat_samples)} valid samples from dataset")

    def _build_valid_samples(self, indices):
        """Build a mapping of dataset indices to their valid subunit indices"""
        valid_samples = {}
        
        for idx in tqdm(indices, desc="Building valid samples"):
            valid_subunits = self._get_valid_subunits(idx)
            if valid_subunits:
                valid_samples[idx] = valid_subunits
        
        return valid_samples
    
    def _get_valid_subunits(self, idx):
        """Get all valid subunit indices for a sample"""
        valid_subunits = []
        try:
            data = self.dataset[idx]
            atoms = data['atoms']
            
            for sub_idx, sub in enumerate(data['labels'].itertuples()):
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa == 20: continue  # Skip unknown amino acids
                
                my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) == 1:
                    valid_subunits.append(sub_idx)
            
        except Exception:
            pass
        
        return valid_subunits

    def generate_grid(self, n, spacing=1):
        """Generate a grid within a given range."""
        start = -spacing
        end = spacing
        coords = torch.linspace(start, end, n)
        
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return grid_coordinates.to(torch.float32)
    
    def __getitem__(self, i):
        idx, sub_idx = self.flat_samples[i]
        data = self.dataset[idx]
        atoms = data['atoms']
        
        # Get the specific subunit
        sub = list(data['labels'].itertuples())[sub_idx]
        _, num, aa = sub.subunit.split('_')
        num, aa = int(num), _amino_acids(aa)
        
        # Get atoms for this subunit
        my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
        ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
        
        # Create a Data object
        grid_data = GridData()
        
        # Extract coordinates and center at CA
        coords = torch.tensor(my_atoms[['x', 'y', 'z']].values, dtype=torch.float32)
        ca_coord = coords[int(ca_idx)]
        coords = coords - ca_coord
        
        # Get atom features
        atom_types = torch.tensor([_element_mapping(e) for e in my_atoms.element], dtype=torch.long)
        
        # Mask residue types for the central residue
        res_types = []
        for res_num, res_name in zip(my_atoms.residue, my_atoms.resname):
            if res_num == num:  # Central residue
                res_types.append(20)  # Mask
            else:
                res_types.append(_amino_acids(res_name))
        res_types = torch.tensor(res_types, dtype=torch.long)
        
        atom_on_bb = torch.tensor([(n in ['N', 'CA', 'C', 'O']) for n in my_atoms.name], dtype=torch.long)
        
        # Physical features
        sasa = torch.zeros(len(my_atoms), dtype=torch.float32)
        charges = torch.zeros(len(my_atoms), dtype=torch.float32)
        
        # Store data
        grid_data.coords = coords
        grid_data.grid_coords = self.grid_coords.clone()
        grid_data.atom_types = atom_types
        grid_data.res_types = res_types
        grid_data.atom_on_bb = atom_on_bb
        grid_data.sasa = sasa
        grid_data.charges = charges
        grid_data.y = torch.tensor(aa, dtype=torch.long)
        grid_data.ca_idx = torch.tensor(int(ca_idx), dtype=torch.long)
        grid_data.num_atoms = coords.size(0)
        grid_data.num_grid_points = self.grid_coords.size(0)
        grid_data.grid_size = self.size
        
        return grid_data

    def __len__(self):
        return len(self.flat_samples)

class ProteinDataset:
    def __init__(self, lmdb_path, split_path_root, batch_size=100, knn=True, radius=2, k=3, size=9, spacing=8, max_samples=None):
        torch_geometric.seed.seed_everything(0)
        self.batch_size = batch_size
 
        self.train_dataset = Protein(
            lmdb_path=lmdb_path,
            split_path=f"{split_path_root}/train_indices.txt",
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing,
            max_samples=max_samples
        )
        self.test_dataset = Protein(
            lmdb_path=lmdb_path,
            split_path=f"{split_path_root}/test_indices.txt",
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing,
            max_samples=max_samples
        )
        self.valid_dataset = Protein(
            lmdb_path=lmdb_path,
            split_path=f"{split_path_root}/val_indices.txt",
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing,
            max_samples=max_samples
        )
    
    def train_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.train_dataset) if distributed else None
        shuffle = not distributed
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, 
                           drop_last=True, sampler=sampler, shuffle=shuffle, num_workers=4)

    def val_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.valid_dataset) if distributed else None
        return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, 
                           drop_last=False, sampler=sampler, shuffle=False, num_workers=4)     

    def test_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.test_dataset) if distributed else None
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, 
                           drop_last=False, sampler=sampler, shuffle=False, num_workers=4)