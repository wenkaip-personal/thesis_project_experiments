import os
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader as PyGDataLoader
import torch_geometric
from pathlib import Path
import json
from torch_cluster import knn, radius
from typing import Any
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

dataroot = os.environ['DATAROOT']
dictroot =  Path(dataroot) / 'protein'

class GridData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'grid_edge_index' in key:
            return torch.tensor([[getattr(self, f'coords').size(0)], [getattr(self, f'grid_coords').size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'grid_edge_index' in key or "edge_index" in key:
            return 1
        else:
            return 0
        
        
class Protein:
    """
    Protein Dataset

    """

    def __init__(self, partition, radius=4.5, k=2, knn=True, size=9, spacing=8):
        with open(str(dictroot / f'{partition}_dict.json'), 'r') as f:
            self.dict = json.load(f)
        self.knn = knn
        self.radius = radius
        self.k = k
        self.grid_coords = self.generate_grid(n=size, spacing=spacing)
        self.size = size
        self.radius_table = torch.tensor([1.7, 1.45, 1.37, 1.7])

    def generate_grid(self, n, spacing=1):
        """
        Generate a grid within a given range.
        
        Parameters:
        - n (int): The size of the grid along one dimension. This will create a n x n x n grid.
        - start (float): The starting value of the range.
        - end (float): The ending value of the range.
        
        Returns:
        - grid_coordinates (Tensor): The coordinates of the grid points.
        """
        start = -spacing
        end = spacing
        # 在给定范围内创建均匀的间距坐标
        coords = torch.linspace(start, end, n)
        
        xx, yy, zz = torch.meshgrid(coords, coords, coords)
        
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return grid_coordinates.to(torch.float64)
    
    def __getitem__(self, i):
        data_path = self.dict[str(i)]
        coords_path = Path(data_path).parent.parent / "atom_coords.pt"
        data = torch.load(data_path)
        coords = torch.load(str(coords_path))[data.node_sub_index]
        cb_coords = coords[data.cb_index]
        coords = coords - cb_coords
        coords = coords@data.rot.T
        data.coords = coords.to(torch.float64)

        data.grid_coords = self.grid_coords
        row_1, col_1 = knn(data.coords, self.grid_coords, k=self.k)
        row_2, col_2 = knn(self.grid_coords, data.coords, k=self.k)

        edge_index_knn = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )

        row_1, col_1 = radius(data.coords, self.grid_coords, r=4)
        row_2, col_2 = radius(self.grid_coords, data.coords, r=4)
        edge_index_radius = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )
        edge_index = torch.cat((edge_index_knn, edge_index_radius), dim=-1)

        edge_index = torch_geometric.utils.coalesce(edge_index)
        data.grid_edge_index = edge_index
        grid_data = GridData()
        grid_data = grid_data.from_dict(data.to_dict())
        grid_data.grid_size = self.size
        grid_data.res_types[data.cb_index] = 20
        return grid_data

    def __len__(self):
        return len(self.dict)

class ProteinInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root=os.environ['DATAROOT'],
        transform=None,
        pre_transform=None,
        pre_filter=None,
        partition="train",
        radius=2,
        k=2,
        knn=True,
        size=9,
        spacing=8
    ):
        self.partition = partition
        self.k=k
        self.knn=knn
        self.size=size
        self.spacing=spacing
        self.radius = radius
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.partition}_data.pt"]

    def process(self):
        self.dataset = Protein(
            partition=self.partition, radius=self.radius, k=self.k, 
            knn=self.knn, size=self.size, 
            spacing=self.spacing
        )
        # Read data into huge `Data` list.
        data_list = []
        for graph in tqdm(self.dataset, total=self.dataset.__len__()):
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinDataset:
    def __init__(self, batch_size=100, knn=True, radius=2, k=3, size=9, spacing=8):
        torch_geometric.seed.seed_everything(0)
        self.batch_size = batch_size
 
        self.train_dataset = Protein(
            partition='train',
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing
        )
        self.test_dataset = Protein(
            partition='test',
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing
        )
        self.valid_dataset = Protein(
            partition='val',
            knn=knn,
            radius=radius,
            k=k,
            size=size,
            spacing=spacing
        )
    
    def train_loader(self):
        split = "train"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.train_dataset) if distributed else None)
        shuffle = True if split == 'train' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=24)

    def val_loader(self):
        split = "valid"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.valid_dataset) if distributed else None)
        shuffle = True if split == 'valid' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=24)     

    def test_loader(self):
        split = "test"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.test_dataset) if distributed else None)
        shuffle = True if split == 'test' and not distributed else False
        drop_last = split == 'train'
        return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, num_workers=24)