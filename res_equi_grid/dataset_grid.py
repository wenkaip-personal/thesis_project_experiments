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

# Element and amino acid mappings from dataset.py
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
        if 'grid_edge_index' in key:
            # Return a scalar value - the maximum number of nodes
            # This ensures all indices will be properly incremented without overlap
            return max(self.coords.size(0), self.grid_coords.size(0))
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'grid_edge_index' in key or "edge_index" in key:
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
            
        # Filter out invalid samples
        # self.idx = self._filter_valid_samples(self.idx)
        
        if max_samples is not None:
            self.idx = self.idx[:max_samples]
            
        self.knn = knn
        self.radius = radius
        self.k = k
        self.grid_coords = self.generate_grid(n=size, spacing=spacing)
        self.size = size
        self.radius_table = torch.tensor([1.7, 1.45, 1.37, 1.7])
        
        print(f"Loaded {len(self.idx)} valid samples from dataset")

    def _filter_valid_samples(self, indices):
        """Filter out samples that don't have valid subunits"""
        valid_indices = []
        
        for idx in tqdm(indices, desc="Filtering valid samples"):
            if self._is_valid_sample(idx):
                valid_indices.append(idx)
        
        return valid_indices
    
    def _is_valid_sample(self, idx):
        """Check if a sample has at least one valid subunit"""
        try:
            data = self.dataset[idx]
            atoms = data['atoms']
            
            for sub in data['labels'].itertuples():
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa == 20: continue  # Skip unknown amino acids
                
                my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) == 1:
                    return True
            
            return False
        except Exception:
            return False

    def generate_grid(self, n, spacing=1):
        """
        Generate a grid within a given range.
        
        Parameters:
        - n (int): The size of the grid along one dimension. This will create a n x n x n grid.
        - spacing (float): Controls the range of the grid (-spacing to +spacing)
        
        Returns:
        - grid_coordinates (Tensor): The coordinates of the grid points.
        """
        start = -spacing
        end = spacing
        # Create evenly spaced coordinates within the given range
        coords = torch.linspace(start, end, n)
        
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return grid_coordinates.to(torch.float64)
    
    def __getitem__(self, i):
        data = self.dataset[self.idx[i]]
        atoms = data['atoms']
        
        # Process for one subunit in data['labels']
        # We're selecting the first valid subunit for simplicity
        for sub in data['labels'].itertuples():
            _, num, aa = sub.subunit.split('_')
            num, aa = int(num), _amino_acids(aa)
            if aa == 20: continue  # Skip unknown amino acids
            
            # Get atoms for this subunit
            my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
            ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
            if len(ca_idx) != 1: continue
            
            # Create a Data object to store the protein information
            grid_data = GridData()
            
            # Extract coordinates and prepare them for the grid
            coords = torch.tensor(my_atoms[['x', 'y', 'z']].values, dtype=torch.float64)
            ca_coord = coords[int(ca_idx)]
            coords = coords - ca_coord  # Center at CA
            
            # Get atom features
            atom_types = torch.tensor([_element_mapping(e) for e in my_atoms.element], dtype=torch.long)
            res_types = torch.tensor([_amino_acids(r) for r in my_atoms.resname], dtype=torch.long)
            atom_on_bb = torch.tensor([(n in ['N', 'CA', 'C', 'O']) for n in my_atoms.name], dtype=torch.long)
            
            # Physical features (placeholder values, replace with actual values if available)
            sasa = torch.zeros(len(my_atoms), dtype=torch.float32)
            charges = torch.zeros(len(my_atoms), dtype=torch.float32)
            
            # Add to grid data
            grid_data.coords = coords
            grid_data.grid_coords = self.grid_coords
            grid_data.atom_types = atom_types
            grid_data.res_types = res_types
            grid_data.atom_on_bb = atom_on_bb
            grid_data.sasa = sasa
            grid_data.charges = charges
            grid_data.y = torch.tensor(aa, dtype=torch.long)  # Target label
            grid_data.cb_index = torch.tensor(int(ca_idx), dtype=torch.long)  # CA index
            
            # Create edges between atoms and grid points
            row_1, col_1 = knn(coords, self.grid_coords, k=self.k)
            row_2, col_2 = knn(self.grid_coords, coords, k=self.k)

            edge_index_knn = torch.stack(
                (torch.cat((col_1, row_2)),
                torch.cat((row_1, col_2)))
            )

            row_1, col_1 = radius(coords, self.grid_coords, r=4)
            row_2, col_2 = radius(self.grid_coords, coords, r=4)
            edge_index_radius = torch.stack(
                (torch.cat((col_1, row_2)),
                torch.cat((row_1, col_2)))
            )
            edge_index = torch.cat((edge_index_knn, edge_index_radius), dim=-1)

            edge_index = torch_geometric.utils.coalesce(edge_index)
            grid_data.grid_edge_index = edge_index
            grid_data.grid_size = self.size

            # Explicit node counts for atoms and grid points
            grid_data.num_nodes = self.grid_coords.size(0) + coords.size(0)
            grid_data.num_atoms = coords.size(0)  # Number of atoms
            grid_data.num_grid_points = self.grid_coords.size(0)  # Number of grid points
            grid_data.is_atom_mask = torch.cat((torch.ones(coords.size(0)), torch.zeros(self.grid_coords.size(0))))

            print(f"Number of edges: {edge_index.shape[1]}")
            print(f"Edge index range: atoms [0, {coords.size(0)-1}], grid [{coords.size(0)}, {coords.size(0) + self.grid_coords.size(0) - 1}]")
            print(f"Actual edge indices - min: {edge_index.min()}, max: {edge_index.max()}")
            
            return grid_data

    def __len__(self):
        return len(self.idx)

class ProteinInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        lmdb_path=None,
        split_path=None,
        radius=2,
        k=2,
        knn=True,
        size=9,
        spacing=8
    ):
        self.lmdb_path = lmdb_path
        self.split_path = split_path
        self.k = k
        self.knn = knn
        self.size = size
        self.spacing = spacing
        self.radius = radius
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        self.dataset = Protein(
            lmdb_path=self.lmdb_path, 
            split_path=self.split_path,
            radius=self.radius, 
            k=self.k, 
            knn=self.knn, 
            size=self.size, 
            spacing=self.spacing
        )
        # Read data into huge `Data` list.
        data_list = []
        for i in tqdm(range(len(self.dataset)), total=self.dataset.__len__()):
            try:
                graph = self.dataset[i]
                data_list.append(graph)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
        split = "train"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.train_dataset) if distributed else None)
        shuffle = True if split == 'train' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=4)

    def val_loader(self):
        split = "valid"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.valid_dataset) if distributed else None)
        shuffle = True if split == 'valid' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=4)     

    def test_loader(self):
        split = "test"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.test_dataset) if distributed else None)
        shuffle = True if split == 'test' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=4)