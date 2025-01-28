import numpy as np
import torch
from torch.utils.data import Dataset

class NBodyTransformerDataset(Dataset):
    """Dataset for N-body system specifically formatted for transformer architecture"""
    
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer"):
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
            
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.data = self.load()

    def load(self):
        # Load raw data
        loc = np.load('../n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load('../n_body_system/dataset/vel_' + self.sufix + '.npy')
        charges = np.load('../n_body_system/dataset/charges_' + self.sufix + '.npy')

        # Preprocess data
        loc, vel, charges = self.preprocess(loc, vel, charges)
        return loc, vel, charges

    def preprocess(self, loc, vel, charges):
        # Convert to torch and swap dimensions 
        loc = torch.tensor(loc, dtype=torch.float32).transpose(2, 3)
        vel = torch.tensor(vel, dtype=torch.float32).transpose(2, 3)
        charges = torch.tensor(charges, dtype=torch.float32)
        
        # Ensure charges has shape [n_samples, n_nodes]
        if len(charges.shape) == 1:
            n_nodes = loc.size(1)
            charges = charges.view(-1, n_nodes)
        
        # Limit number of samples
        loc = loc[0:self.max_samples]
        vel = vel[0:self.max_samples] 
        charges = charges[0:self.max_samples]

        return loc, vel, charges

    def __getitem__(self, i):
        loc, vel, charges = self.data
        
        # Select appropriate frames based on dataset
        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[i, frame_0], vel[i, frame_0], charges[i], loc[i, frame_T]

    def __len__(self):
        return len(self.data[0])

    def get_n_nodes(self):
        return self.data[0].size(1)