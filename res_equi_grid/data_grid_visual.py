import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset_grid import GridRESDataset
from torch_geometric.loader import DataLoader
import os

# Create a dataset with a small number of samples for visualization
dataset = GridRESDataset(
    '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/',
    '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/train_indices.txt',
    grid_size=9,
    spacing=2.0,
    k=3,
    max_samples=10  # Limit to 10 samples for quick visualization
)

# Create a dataloader with batch_size=2 as requested
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Get a single batch
batch = next(iter(dataloader))

# Visualization function
def visualize_sample(data, sample_idx=0):
    """
    Visualize a single sample from the batch
    
    Args:
        data: Batched data
        sample_idx: Index of the sample in the batch to visualize
    """
    # Get mask for the current sample
    mask = data.batch == sample_idx
    
    # Extract atom positions and grid positions for this sample
    atom_pos = data.x[mask].cpu().numpy()
    
    # Get the starting index for grid points
    atom_count = mask.sum().item()
    grid_size = data.grid_size
    grid_count = grid_size**3
    
    # Create figures
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Atom positions
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2], c='blue', alpha=0.7, s=30)
    
    # Highlight CA atom
    ca_idx = data.ca_idx[sample_idx].item()
    ca_pos = atom_pos[ca_idx]
    ax1.scatter(ca_pos[0], ca_pos[1], ca_pos[2], c='red', s=100, marker='*')
    
    ax1.set_title(f'Atom Positions (Sample {sample_idx})\nAtom count: {atom_count}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Grid positions
    grid_pos = data.grid_coords.cpu().numpy()
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(grid_pos[:, 0], grid_pos[:, 1], grid_pos[:, 2], c='green', alpha=0.3, s=10)
    ax2.set_title(f'Grid Points\nGrid size: {grid_size}x{grid_size}x{grid_size}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot 3: Combined view with edges
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2], c='blue', alpha=0.7, s=30, label='Atoms')
    ax3.scatter(grid_pos[:, 0], grid_pos[:, 1], grid_pos[:, 2], c='green', alpha=0.3, s=10, label='Grid')
    ax3.scatter(ca_pos[0], ca_pos[1], ca_pos[2], c='red', s=100, marker='*', label='CA Atom')
    
    # Extract edges for this sample
    edge_index = data.grid_edge_index.cpu().numpy()
    
    # Filter edges for this sample
    valid_source_mask = edge_index[0] < atom_count
    valid_edges = edge_index[:, valid_source_mask]
    
    # Counter for edges plotted
    edges_plotted = 0
    
    # Plot ALL edges (not just a subset)
    for i in range(valid_edges.shape[1]):
        src_idx = valid_edges[0, i]
        tgt_idx = valid_edges[1, i]
        
        # Only plot if target is a grid point
        if tgt_idx >= atom_count:
            src_pos = atom_pos[src_idx]
            tgt_pos = grid_pos[tgt_idx - atom_count]  # Adjust index for grid points
            ax3.plot([src_pos[0], tgt_pos[0]], 
                     [src_pos[1], tgt_pos[1]], 
                     [src_pos[2], tgt_pos[2]], 'gray', alpha=0.05)  # Reduced alpha for clarity
            edges_plotted += 1
    
    ax3.set_title(f'Atoms + Grid + Connections\nEdges plotted: {edges_plotted}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    return fig

# Print statistics about the batch
print(f"Batch size: {batch.batch.max().item() + 1}")
print(f"Total atoms: {batch.x.shape[0]}")
print(f"Grid size per sample: {batch.grid_size}x{batch.grid_size}x{batch.grid_size}")
print(f"Total grid points: {batch.grid_coords.shape[0]}")
print(f"Grid edge index shape: {batch.grid_edge_index.shape}")
print(f"Labels: {batch.label}")

# Visualize the first sample
fig1 = visualize_sample(batch, sample_idx=0)
plt.savefig('sample_0_visualization_all_edges.png', dpi=300, bbox_inches='tight')

# Visualize the second sample
fig2 = visualize_sample(batch, sample_idx=1)
plt.savefig('sample_1_visualization_all_edges.png', dpi=300, bbox_inches='tight')

# Edge statistics
edge_index = batch.grid_edge_index.cpu().numpy()
print(f"\nEdge Statistics:")
print(f"Total edges: {edge_index.shape[1]}")
print(f"Average connections per atom: {edge_index.shape[1] / batch.x.shape[0]:.2f}")
print(f"Max source index: {edge_index[0].max()}")
print(f"Max target index: {edge_index[1].max()}")