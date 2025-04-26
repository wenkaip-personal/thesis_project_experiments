import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import RESDataset
from model import ResOrientedMP

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

def test_rotation_invariance(model, sample, device, num_rotations=5):
    """
    Test whether the model produces rotation-invariant features
    
    Args:
        model: The neural network model
        sample: A protein micro-environment sample
        device: The device to run on (CPU/GPU)
        num_rotations: Number of random rotations to test
    
    Returns:
        List of feature differences between original and rotated samples
    """
    # Move sample to device
    sample = sample.to(device)
    
    # Get original features (before final classification layer)
    model.eval()
    with torch.no_grad():
        original_features = model.get_node_features(sample.atoms, sample.x, sample.edge_index, sample)
        original_output = model(sample.atoms, sample.x, sample.edge_index, sample)
    
    diffs = []
    for i in range(num_rotations):
        # Create random rotation matrix
        rotation = random_rotation_matrix()
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=device)
        
        # Create a copy of the sample with rotated coordinates
        rotated_sample = sample.clone()
        rotated_sample.x = torch.matmul(sample.x, rotation_tensor.t())
        
        # Get features from rotated sample
        with torch.no_grad():
            rotated_features = model.get_node_features(
                rotated_sample.atoms, rotated_sample.x, rotated_sample.edge_index, rotated_sample
            )
            rotated_output = model(
                rotated_sample.atoms, rotated_sample.x, rotated_sample.edge_index, rotated_sample
            )
        
        # Calculate difference in features
        feature_diff = torch.norm(original_features - rotated_features).item()
        output_diff = torch.norm(original_output - rotated_output).item()
        
        print(f"Rotation {i+1}:")
        print(f"  Feature difference: {feature_diff:.6f}")
        print(f"  Output difference: {output_diff:.6f}")
        
        diffs.append(feature_diff)
    
    return diffs

def test_global_orientation_invariance(model, sample, device, num_rotations=5):
    """
    Test whether the global orientation transformation produces rotation-invariant features
    
    Args:
        model: The neural network model
        sample: A protein micro-environment sample
        device: The device to run on (CPU/GPU)
        num_rotations: Number of random rotations to test
    
    Returns:
        List of feature differences between original and rotated samples
    """
    # Move sample to device
    sample = sample.to(device)
    
    # Get original transformed coordinates and orientations
    model.eval()
    with torch.no_grad():
        transformed_x, global_orientations = model.global_orientation_transform(
            sample.atoms, sample.x, sample.edge_index, sample
        )
        output, _ = model.forward_with_global_orientation(
            sample.atoms, sample.x, sample.edge_index, sample
        )
    
    diffs = []
    orientation_diffs = []
    coord_diffs = []
    
    for i in range(num_rotations):
        # Create random rotation matrix
        rotation = random_rotation_matrix()
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=device)
        
        # Create a copy of the sample with rotated coordinates
        rotated_sample = sample.clone()
        rotated_sample.x = torch.matmul(sample.x, rotation_tensor.t())
        
        # Get transformed coordinates and orientations from rotated sample
        with torch.no_grad():
            rotated_transformed_x, rotated_global_orientations = model.global_orientation_transform(
                rotated_sample.atoms, rotated_sample.x, rotated_sample.edge_index, rotated_sample
            )
            rotated_output, _ = model.forward_with_global_orientation(
                rotated_sample.atoms, rotated_sample.x, rotated_sample.edge_index, rotated_sample
            )
        
        # Calculate differences
        # 1. Coordinate difference after global orientation transform
        coord_diff = torch.norm(transformed_x - rotated_transformed_x).item()
        coord_diffs.append(coord_diff)
        
        # 2. Global orientation difference (need to consider that they might differ by rotation)
        orientation_diff = torch.norm(global_orientations - torch.matmul(rotation_tensor, rotated_global_orientations)).item()
        orientation_diffs.append(orientation_diff)
        
        # 3. Output difference
        output_diff = torch.norm(output - rotated_output).item()
        diffs.append(output_diff)
        
        print(f"Global Orientation Test - Rotation {i+1}:")
        print(f"  Transformed coordinate difference: {coord_diff:.6f}")
        print(f"  Orientation matrix difference: {orientation_diff:.6f}")
        print(f"  Output difference: {output_diff:.6f}")
    
    return diffs, orientation_diffs, coord_diffs

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/'
    split_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/test_indices.txt'
    
    # Load a single sample for testing
    dataset = RESDataset(data_path, split_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    sample = next(iter(loader))
    
    # Initialize model (or load pre-trained model)
    model = ResOrientedMP(
        in_node_nf=9,      # One-hot encoded atoms
        hidden_nf=128,     # Hidden dimension
        out_node_nf=20,    # Number of amino acid classes
        in_edge_nf=16,     # RBF edge features
        n_layers=4         # Number of message passing layers
    ).to(device)
    
    # Optional: Load pre-trained weights if available
    # model.load_state_dict(torch.load('/path/to/model.pt'))
    
    # Run standard rotation invariance test
    print("Testing standard model rotation invariance...")
    diffs = test_rotation_invariance(model, sample, device)
    print("\nSummary:")
    print(f"Average feature difference across rotations: {sum(diffs)/len(diffs):.6f}")
    print(f"Maximum feature difference: {max(diffs):.6f}")
    
    # Run global orientation invariance test
    print("\nTesting global orientation rotation invariance...")
    output_diffs, orientation_diffs, coord_diffs = test_global_orientation_invariance(model, sample, device)
    print("\nGlobal Orientation Summary:")
    print(f"Average output difference: {sum(output_diffs)/len(output_diffs):.6f}")
    print(f"Average orientation difference: {sum(orientation_diffs)/len(orientation_diffs):.6f}")
    print(f"Average coordinate difference: {sum(coord_diffs)/len(coord_diffs):.6f}")
    
    # If the global orientation is working correctly, we should see very small differences in coordinates
    # after transformation, despite the input coordinates being rotated

if __name__ == "__main__":
    main()