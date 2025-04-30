import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import RESDataset
from model import EquivariantResModel

def random_rotation_matrix():
    """Generate a random 3D rotation matrix"""
    # Random rotation matrix via QR decomposition
    H = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det=1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def measure_equivariance_error(model, dataset, device, num_samples=10, num_rotations=5):
    """
    Measure equivariance error by comparing outputs for original and rotated inputs
    
    Args:
        model: The neural network model
        dataset: Dataset containing samples
        device: The device to run on (CPU/GPU)
        num_samples: Number of samples to test
        num_rotations: Number of random rotations to test per sample
        
    Returns:
        Average equivariance error
    """
    # Initialize data loader
    loader = DataLoader(dataset, batch_size=1)
    
    # Initialize error metrics
    output_errors = []
    
    # Test equivariance on multiple samples
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= num_samples:
                break
                
            # Move sample to device
            sample = sample.to(device)
            
            # Get original output
            original_output = model(sample.atoms, sample.x, sample.edge_index, sample)
            
            # Test multiple rotations
            for j in range(num_rotations):
                # Create random rotation matrix
                rotation = random_rotation_matrix()
                rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=device)
                
                # Create rotated sample
                rotated_x = torch.matmul(sample.x, rotation_tensor.t())
                
                # Get output from rotated input
                rotated_output = model(sample.atoms, rotated_x, sample.edge_index, sample)
                
                # Calculate error
                error = torch.norm(original_output - rotated_output).item()
                output_errors.append(error)
                
                print(f"Sample {i+1}, Rotation {j+1}: Error = {error:.6f}")
    
    # Calculate average errors
    avg_output_error = sum(output_errors) / len(output_errors)
    
    print(f"\nAverage output error: {avg_output_error:.6f}")
    
    return avg_output_error

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/'
    split_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/test_indices.txt'
    
    # Load dataset
    dataset = RESDataset(data_path, split_path)
    
    # Initialize model
    model = EquivariantResModel(
        in_node_nf=9,      # Number of atom types
        hidden_nf=128,     # Hidden dimension
        out_node_nf=20,    # Number of amino acid classes
        edge_nf=16,        # Edge feature dimension
        n_layers=4,        # Number of message passing layers
        device=device
    ).to(device)
    
    # Optional: Load pre-trained weights if available
    # model.load_state_dict(torch.load('path/to/model.pt'))
    
    # Measure equivariance error
    error = measure_equivariance_error(model, dataset, device)
    
    print(f"Equivariance test completed. Average error: {error:.6f}")

if __name__ == "__main__":
    main()