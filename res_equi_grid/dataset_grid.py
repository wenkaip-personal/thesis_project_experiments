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
from torch.utils.data import IterableDataset

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

class Protein(IterableDataset):
    """
    Protein Dataset based on ATOM3D RES dataset
    """

    def __init__(self, lmdb_path, split_path=None, radius=4.5, k=2, knn=True, size=5, spacing=2, max_samples=None):
        self.dataset = LMDBDataset(lmdb_path)
        
        if split_path is not None:
            self.idx = list(map(int, open(split_path).read().split()))
        else:
            self.idx = list(range(len(self.dataset)))
        
        if max_samples is not None:
            self.idx = self.idx[:max_samples]
            
        self.knn = knn
        self.radius = radius
        self.k = k
        self.grid_coords = self.generate_grid(n=size, spacing=spacing)
        self.size = size
        self.radius_table = torch.tensor([1.7, 1.45, 1.37, 1.7])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))), shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end], shuffle=True)
        return gen
    
    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: 
            random.shuffle(indices)
            
        for idx in indices:
            data = self.dataset[self.idx[idx]]
            atoms = data['atoms']
            
            for sub in data['labels'].itertuples():
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa == 20: continue
                
                my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) != 1: continue
                
                # 获取中心氨基酸的C、N、Calpha原子坐标
                c_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'C'))[0]
                n_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'N'))[0]
                
                # 确保能找到所有三个原子
                if len(c_idx) != 1 or len(n_idx) != 1:
                    continue
                    
                # 获取三个原子的坐标
                ca_coord = my_atoms.iloc[int(ca_idx)][['x', 'y', 'z']].values
                c_coord = my_atoms.iloc[int(c_idx)][['x', 'y', 'z']].values
                n_coord = my_atoms.iloc[int(n_idx)][['x', 'y', 'z']].values
                # 计算局部坐标系的三个轴
                # z轴：从Calpha指向C
                z_axis = c_coord - ca_coord
                z_axis = z_axis / np.linalg.norm(z_axis)
                z_axis = z_axis.astype(np.float64)
                
                # x轴：垂直于Calpha-C-N平面
                n_to_ca = ca_coord - n_coord
                n_to_ca = n_to_ca.astype(np.float64)
                x_axis = np.cross(n_to_ca, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                
                # y轴：由x轴和z轴叉乘得到
                y_axis = np.cross(z_axis, x_axis)
                
                # 构建旋转矩阵（frame）
                rotation_matrix = np.array([x_axis, y_axis, z_axis])
                
                # 所有原子坐标转换到局部坐标系
                coords = my_atoms[['x', 'y', 'z']].values
                centered_coords = coords - ca_coord  # 先将坐标中心移到Calpha
                rotated_coords = np.dot(centered_coords, rotation_matrix.T)
                rotated_coords = rotated_coords.astype(np.float64)
                
                # 转换为PyTorch张量
                # coords = torch.tensor(rotated_coords, dtype=torch.float64)
                centered_coords = centered_coords - centered_coords.mean(0)
                coords = torch.tensor(centered_coords, dtype=torch.float64)
                
                # Create grid data for this subunit
                grid_data = GridData()
                
                atom_types = torch.tensor([_element_mapping(e) for e in my_atoms.element], dtype=torch.long)
                
                res_types = []
                for idx, (res_num, res_name) in enumerate(zip(my_atoms.residue, my_atoms.resname)):
                    if res_num == num:
                        res_types.append(20)
                    else:
                        res_types.append(_amino_acids(res_name))
                res_types = torch.tensor(res_types, dtype=torch.long)
                
                atom_on_bb = torch.tensor([(n in ['N', 'CA', 'C', 'O']) for n in my_atoms.name], dtype=torch.long)
                sasa = torch.zeros(len(my_atoms), dtype=torch.float32)
                charges = torch.zeros(len(my_atoms), dtype=torch.float32)
                grid_data.coords = coords
                grid_data.grid_coords = self.grid_coords
                grid_data.atom_types = atom_types
                grid_data.res_types = res_types
                grid_data.atom_on_bb = atom_on_bb
                grid_data.sasa = sasa
                grid_data.charges = charges
                grid_data.y = torch.tensor(aa, dtype=torch.long)
                grid_data.cb_index = torch.tensor(int(ca_idx), dtype=torch.long)
                grid_data.grid_size = self.size
                grid_data.num_nodes = self.grid_coords.size(0) + coords.size(0)
                grid_data.num_atoms = coords.size(0)
                grid_data.num_grid_points = self.grid_coords.size(0)
                grid_data.is_atom_mask = torch.cat((torch.ones(coords.size(0)), torch.zeros(self.grid_coords.size(0))))
                
                yield grid_data

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
        coords = torch.linspace(start, end, n)
        
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return grid_coordinates.to(torch.float64)

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
        drop_last = split == 'train'
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, num_workers=0)

    def val_loader(self):
        split = "valid"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.valid_dataset) if distributed else None)
        drop_last = split == 'train'
        return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, num_workers=0)     

    def test_loader(self):
        split = "test"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.test_dataset) if distributed else None)
        drop_last = split == 'train'
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, num_workers=0)