import torch, random, math
import numpy as np
from atom3d.datasets import LMDBDataset
from torch.utils.data import IterableDataset
import torch_cluster, torch_geometric

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
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v

class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                    dtype=torch.float32, device=self.device)
        atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                        dtype=torch.long, device=self.device)

        edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

        edge_s, edge_v = _edge_features(coords, edge_index, 
                            D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

        return torch_geometric.data.Data(x=coords, atoms=atoms,
                    edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)

class RESDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D RES dataset.
    
    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon, 
    and all structural attributes as described in BaseTransform.
    
    Excludes hydrogen atoms.
    
    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    '''
    def __init__(self, lmdb_dataset, split_path):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.idx = list(map(int, open(split_path).read().split()))
        self.transform = BaseTransform()
        
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
        if shuffle: random.shuffle(indices)
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
                    
                graph = self.transform(my_atoms)
                graph.label = aa
                graph.ca_idx = int(ca_idx)
                yield graph

    def __len__(self):
        return len(self.dataset)