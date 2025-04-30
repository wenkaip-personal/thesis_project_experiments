import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
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
        Average equivariance errors
    """
    # Initialize data loader
    loader = DataLoader(dataset, batch_size=1)
    
    # Initialize error metrics
    output_errors = []
    orientation_errors = []
    node_feature_errors = []
    
    # Test equivariance on multiple samples
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= num_samples:
                break
                
            # Move sample to device
            sample = sample.to(device)
            
            # Generate initial atom embeddings
            initial_embeddings = model.atom_embedding(sample.atoms)
            
            # Get original node features after message passing
            original_x_centered = sample.x
            if hasattr(sample, 'batch'):
                batch_idx = sample.batch
                original_x_centered = sample.x - global_mean_pool(sample.x, batch_idx)[batch_idx]
                
            original_orientations = model.get_orientations(initial_embeddings, original_x_centered, 
                                                         sample.edge_index, sample.edge_s)
            
            # Process through message passing to get node features
            original_node_features = model.message_passing(initial_embeddings, sample.x, 
                                                         sample.edge_index, original_orientations, 
                                                         sample.edge_s)
            
            # Get original output
            original_output = model.output_mlp(original_node_features[sample.ca_idx] 
                                            if hasattr(sample, 'ca_idx') else original_node_features)
            
            # Print original feature values
            print(f"\n=== Sample {i+1} ===")
            print(f"Original coordinates (first 3 nodes):")
            print(sample.x[:3])
            
            print(f"Original node embeddings (first node, first 5 values):")
            print(initial_embeddings[0, :5])
            
            print(f"Original node features after message passing (first node, first 5 values):")
            print(original_node_features[0, :5])
            
            print(f"Original learned orientations (first node):")
            print(original_orientations[0])
            
            print(f"Original model output (first 5 classes):")
            print(original_output[0, :5])
            
            # Test multiple rotations
            for j in range(num_rotations):
                # Create random rotation matrix
                rotation = random_rotation_matrix()
                rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=device)
                
                # Print rotation matrix
                print(f"\n--- Rotation {j+1} ---")
                print(f"Rotation matrix:")
                print(rotation_tensor)
                
                # Create rotated sample
                rotated_x = torch.matmul(sample.x, rotation_tensor.t())
                
                # Print rotated coordinates
                print(f"Rotated coordinates (first 3 nodes):")
                print(rotated_x[:3])
                
                # Get rotated node features
                rotated_x_centered = rotated_x
                if hasattr(sample, 'batch'):
                    batch_idx = sample.batch
                    rotated_x_centered = rotated_x - global_mean_pool(rotated_x, batch_idx)[batch_idx]
                    
                rotated_orientations = model.get_orientations(initial_embeddings, rotated_x_centered, 
                                                           sample.edge_index, sample.edge_s)
                
                # Process through message passing to get node features
                rotated_node_features = model.message_passing(initial_embeddings, rotated_x, 
                                                           sample.edge_index, rotated_orientations, 
                                                           sample.edge_s)
                
                # Get final output
                rotated_output = model.output_mlp(rotated_node_features[sample.ca_idx] 
                                               if hasattr(sample, 'ca_idx') else rotated_node_features)
                
                # Print rotated feature values
                print(f"Rotated node features after message passing (first node, first 5 values):")
                print(rotated_node_features[0, :5])
                
                print(f"Rotated learned orientations (first node):")
                print(rotated_orientations[0])
                
                print(f"Rotated model output (first 5 classes):")
                print(rotated_output[0, :5])
                
                # Calculate errors
                output_error = torch.norm(original_output - rotated_output).item()
                output_errors.append(output_error)
                
                # Node feature error
                node_feature_error = torch.norm(original_node_features - rotated_node_features).item()
                node_feature_errors.append(node_feature_error)
                
                # Orientation error
                expected_orientation = torch.matmul(rotation_tensor, original_orientations[0])
                orientation_error = torch.norm(expected_orientation - rotated_orientations[0]).item()
                orientation_errors.append(orientation_error)
                
                # Print feature differences
                print(f"Node feature difference (first node):")
                print(original_node_features[0] - rotated_node_features[0])
                
                print(f"Output error = {output_error:.6f}")
                print(f"Node feature error = {node_feature_error:.6f}")
                print(f"Orientation error = {orientation_error:.6f}")
    
    # Calculate average errors
    avg_output_error = sum(output_errors) / len(output_errors)
    avg_node_feature_error = sum(node_feature_errors) / len(node_feature_errors)
    avg_orientation_error = sum(orientation_errors) / len(orientation_errors)
    
    print(f"\nAverage output error: {avg_output_error:.6f}")
    print(f"Average node feature error: {avg_node_feature_error:.6f}")
    print(f"Average orientation error: {avg_orientation_error:.6f}")
    
    return avg_output_error, avg_node_feature_error, avg_orientation_error

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
    output_error, node_feature_error, orientation_error = measure_equivariance_error(
        model, dataset, device)
    
    print(f"Equivariance test completed.")
    print(f"Output error: {output_error:.6f}")
    print(f"Node feature error: {node_feature_error:.6f}")
    print(f"Orientation error: {orientation_error:.6f}")

if __name__ == "__main__":
    main()