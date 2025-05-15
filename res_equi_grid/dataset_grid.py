import os
import torch
import math
import random
import numpy as np
from atom3d.datasets import LMDBDataset
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Data, Batch
import torch_cluster
from tqdm import tqdm

# Mapping functions
_element_mapping = lambda x: {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'CL': 6, 'P': 7
}.get(x, 8)

_amino_acids = lambda x: {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 
    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 
    'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}.get(x, 20)

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    '''
    Create edge features from atomic coordinates and edge indices.
    '''
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v

class GridTransform:
    '''
    Transform for converting atomic structures to grid-based representations.
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, grid_size=9, spacing=2.0, 
                 k=3, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.grid_size = grid_size
        self.spacing = spacing
        self.k = k
        self.grid_coords = self._generate_grid(grid_size, spacing)
        
    def _generate_grid(self, grid_size, spacing):
        '''
        Generate a regular grid with specified size and spacing.
        '''
        # FIX: Adjust grid generation to ensure it covers protein environment better
        start = -spacing * 0.5  # Adjust start to be half of spacing for better centering
        end = spacing * 0.5
        # Create evenly spaced coordinates in the given range
        coords = torch.linspace(start, end, grid_size)
        
        # Create 3D grid using meshgrid
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        # Flatten the grid to get all coordinates
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
        return grid_coordinates
            
    def __call__(self, df):
        '''
        Transform ATOM3D dataframe to grid representation.
        
        :param df: `pandas.DataFrame` of atomic coordinates in ATOM3D format
        :return: `torch_geometric.data.Data` structure with grid representation
        '''
        # Process atomic data
        coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                dtype=torch.float32, device=self.device)
        atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                dtype=torch.long, device=self.device)

        # FIX: Center the coordinates on the CA atom if specified in the transform
        ca_idx = df.index[(df.name == 'CA') & (df.element == 'C')].tolist()
        if len(ca_idx) == 1:
            ca_pos = coords[ca_idx[0]]
            coords = coords - ca_pos
            
        # Create point cloud graph
        edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)
        edge_s, edge_v = _edge_features(coords, edge_index, 
                            D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)
        
        # Add grid information
        grid_coords = self.grid_coords.to(self.device)
        
        # FIX: Simplify connectivity strategy to use only KNN for more consistency
        row_1, col_1 = torch_cluster.knn(coords, grid_coords, k=self.k)
        row_2, col_2 = torch_cluster.knn(grid_coords, coords, k=self.k)
        
        # Combine connections for bidirectional connectivity
        grid_edge_index = torch.stack(
            (torch.cat((col_1, row_2)),
            torch.cat((row_1, col_2)))
        )
        
        # FIX: Add explicit check for empty edges
        if grid_edge_index.numel() == 0:
            # Fallback to simple connections if no edges found
            print("Warning: No grid edges found, falling back to simple connections")
            grid_edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        
        # Create data object
        data = Data(
            x=coords,
            atoms=atoms,
            edge_index=edge_index,
            edge_s=edge_s,
            edge_v=edge_v,
            grid_coords=grid_coords,
            grid_edge_index=grid_edge_index,
            grid_size=self.grid_size
        )
        
        return data

class GridRESDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around an ATOM3D RES dataset
    with grid representation.
    
    On each iteration, returns a `torch_geometric.data.Data` graph with grid representation
    including the attribute `label` encoding the masked residue identity, 
    `ca_idx` for the node index of the alpha carbon, and all structural attributes.
    
    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    :param grid_size: size of the grid (grid_size x grid_size x grid_size)
    :param spacing: spacing between grid points
    :param k: number of nearest neighbors for grid connections
    '''
    def __init__(self, lmdb_dataset, split_path, grid_size=9, spacing=2.0, k=3, max_samples=None):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.idx = list(map(int, open(split_path).read().split()))
        if max_samples is not None:
            self.idx = self.idx[:max_samples]
        self.transform = GridTransform(grid_size=grid_size, spacing=spacing, k=k)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))), 
                      shuffle=True)
        else:  
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end],
                      shuffle=True)
        return gen
    
    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: 
            random.shuffle(indices)
            
        for idx in indices:
            try:
                data = self.dataset[self.idx[idx]]
                atoms = data['atoms']
                
                for sub in data['labels'].itertuples():
                    _, num, aa = sub.subunit.split('_')
                    num, aa = int(num), _amino_acids(aa)
                    if aa == 20:  # Skip unknown amino acids
                        continue
                        
                    # Extract atoms for this subunit
                    my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                    
                    # Find CA atom index
                    ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                    if len(ca_idx) != 1:
                        continue
                    
                    # Apply grid transform
                    graph = self.transform(my_atoms)
                    graph.label = aa
                    graph.ca_idx = int(ca_idx)
                    
                    yield graph
            except Exception as e:
                # FIX: Add better error handling to skip problematic samples
                print(f"Error processing sample {idx}: {str(e)}")
                continue