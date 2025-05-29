import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch_cluster import knn, radius
import sys
import os

# Add the path to import custom modules
sys.path.append('/content/drive/MyDrive/thesis_project/thesis_project_experiments/res_equi_grid/')

# Import required classes from the existing implementation
from dataset_grid import Protein, GridData, _element_mapping, _amino_acids

# Set up the data paths
lmdb_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/'
split_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/train_indices.txt'

# Initialize the dataset with the same parameters as in main_grid.py
dataset = Protein(
    lmdb_path=lmdb_path,
    split_path=split_path,
    radius=4.5,
    k=2,
    knn=True,
    size=9,
    spacing=8,
    max_samples=10  # Load only a few samples for visualization
)

# Get the first valid sample
sample_idx = 0
grid_data = dataset[sample_idx]

# Extract data from the sample
atom_coords = grid_data.coords.numpy()  # Atom positions (centered at CA)
grid_coords = grid_data.grid_coords.numpy()  # Grid positions
atom_types = grid_data.atom_types.numpy()
res_types = grid_data.res_types.numpy()
ca_idx = grid_data.cb_index.item()  # CA atom index

# Element mapping reverse lookup for visualization
element_names = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Other']
amino_acid_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                    'TYR', 'VAL', 'UNK']

# Compute edges using the same logic as in the model
# Convert to torch tensors for edge computation
atom_coords_torch = torch.tensor(atom_coords, dtype=torch.float32)
grid_coords_torch = torch.tensor(grid_coords, dtype=torch.float32)

# Create batch indices (single sample, so all zeros)
atom_batch = torch.zeros(len(atom_coords), dtype=torch.long)
grid_batch = torch.zeros(len(grid_coords), dtype=torch.long)

# Compute KNN edges (k=3 as in model)
row_1, col_1 = knn(atom_coords_torch, grid_coords_torch, k=3, batch_x=atom_batch, batch_y=grid_batch)
row_2, col_2 = knn(grid_coords_torch, atom_coords_torch, k=3, batch_x=grid_batch, batch_y=atom_batch)

edge_index_knn = torch.stack(
    (torch.cat((col_1, row_2)),
     torch.cat((row_1, col_2)))
)

# Compute radius edges (r=4 as in model)
row_1_r, col_1_r = radius(atom_coords_torch, grid_coords_torch, r=4, batch_x=atom_batch, batch_y=grid_batch)
row_2_r, col_2_r = radius(grid_coords_torch, atom_coords_torch, r=4, batch_x=grid_batch, batch_y=atom_batch)

edge_index_radius = torch.stack(
    (torch.cat((col_1_r, row_2_r)),
     torch.cat((row_1_r, col_2_r)))
)

# Combine and coalesce edges
edge_index = torch.cat((edge_index_knn, edge_index_radius), dim=-1)
edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 1. Sample Overview - Show all atoms with their types
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('Sample Protein Structure\n(Colored by Element Type)', fontsize=14)

# Color atoms by element type
colors = plt.cm.tab10(np.linspace(0, 1, 9))
for i, (coord, elem_type) in enumerate(zip(atom_coords, atom_types)):
    color = colors[elem_type]
    size = 100 if i == ca_idx else 50
    marker = 'o' if i == ca_idx else 'o'
    ax1.scatter(coord[0], coord[1], coord[2], c=[color], s=size, marker=marker, alpha=0.8)
    if i == ca_idx:
        ax1.text(coord[0], coord[1], coord[2], 'CA', fontsize=12, fontweight='bold')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'Sample Protein Structure\n(Target: {amino_acid_names[grid_data.y.item()]})', fontsize=14)

# 2. Grid Structure
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_title('Grid Structure (9x9x9)', fontsize=14)
ax2.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='red', s=30, alpha=0.5, marker='s')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# 3. Combined View - Atoms and Grid
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_title('Atoms and Grid Combined', fontsize=14)

# Plot atoms
for i, coord in enumerate(atom_coords):
    size = 150 if i == ca_idx else 80
    ax3.scatter(coord[0], coord[1], coord[2], c='blue', s=size, alpha=0.8)
    if i == ca_idx:
        ax3.text(coord[0], coord[1], coord[2], 'CA', fontsize=10, fontweight='bold')

# Plot grid points
ax3.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='red', s=20, alpha=0.3, marker='s')

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# 4. KNN Edges
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.set_title('K-Nearest Neighbor Edges (k=3)', fontsize=14)

# Plot atoms and grid
ax4.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
           c='blue', s=50, alpha=0.8, label='Atoms')
ax4.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='red', s=20, alpha=0.3, marker='s', label='Grid')

# Draw KNN edges
knn_edges = edge_index_knn.numpy()
num_atoms = len(atom_coords)
for i in range(knn_edges.shape[1]):
    idx1, idx2 = knn_edges[:, i]
    # Determine if edge connects atom to grid or grid to atom
    if idx1 < num_atoms:  # atom to grid
        start = atom_coords[idx1]
        end = grid_coords[idx2]
    else:  # grid to atom
        start = grid_coords[idx1 - num_atoms]
        end = atom_coords[idx2]
    ax4.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            'g-', alpha=0.2, linewidth=0.5)

ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend()

# 5. Radius Edges
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.set_title('Radius Edges (r=4.0)', fontsize=14)

# Plot atoms and grid
ax5.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
           c='blue', s=50, alpha=0.8, label='Atoms')
ax5.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='red', s=20, alpha=0.3, marker='s', label='Grid')

# Draw radius edges
radius_edges = edge_index_radius.numpy()
for i in range(radius_edges.shape[1]):
    idx1, idx2 = radius_edges[:, i]
    # Determine if edge connects atom to grid or grid to atom
    if idx1 < num_atoms:  # atom to grid
        start = atom_coords[idx1]
        end = grid_coords[idx2]
    else:  # grid to atom
        start = grid_coords[idx1 - num_atoms]
        end = atom_coords[idx2]
    ax5.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            'orange', alpha=0.1, linewidth=0.5)

ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.legend()

# 6. All Edges Combined
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.set_title('All Edges (KNN + Radius)', fontsize=14)

# Plot atoms with CA highlighted
for i, coord in enumerate(atom_coords):
    if i == ca_idx:
        ax6.scatter(coord[0], coord[1], coord[2], c='green', s=200, alpha=1.0, 
                   marker='*', label='CA (Central)', edgecolors='black', linewidth=2)
    else:
        ax6.scatter(coord[0], coord[1], coord[2], c='blue', s=50, alpha=0.8)

# Plot grid points
ax6.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='red', s=20, alpha=0.3, marker='s', label='Grid')

# Draw all edges
all_edges = edge_index.numpy()
for i in range(all_edges.shape[1]):
    idx1, idx2 = all_edges[:, i]
    # Determine if edge connects atom to grid or grid to atom
    if idx1 < num_atoms:  # atom to grid
        start = atom_coords[idx1]
        end = grid_coords[idx2]
    else:  # grid to atom
        start = grid_coords[idx1 - num_atoms]
        end = atom_coords[idx2]
    ax6.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            'gray', alpha=0.1, linewidth=0.5)

ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('equi_grid_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"Sample Statistics:")
print(f"- Number of atoms: {len(atom_coords)}")
print(f"- Number of grid points: {len(grid_coords)}")
print(f"- Target amino acid: {amino_acid_names[grid_data.y.item()]}")
print(f"- CA atom index: {ca_idx}")
print(f"- Number of KNN edges: {edge_index_knn.shape[1]}")
print(f"- Number of radius edges: {edge_index_radius.shape[1]}")
print(f"- Total number of edges: {edge_index.shape[1]}")

# Create a second figure for detailed edge analysis
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 5))

# Edge distance distribution
edge_distances = []
for i in range(all_edges.shape[1]):
    idx1, idx2 = all_edges[:, i]
    if idx1 < num_atoms:
        dist = np.linalg.norm(atom_coords[idx1] - grid_coords[idx2])
    else:
        dist = np.linalg.norm(grid_coords[idx1 - num_atoms] - atom_coords[idx2])
    edge_distances.append(dist)

ax7.hist(edge_distances, bins=50, alpha=0.7, color='purple')
ax7.set_xlabel('Edge Distance')
ax7.set_ylabel('Count')
ax7.set_title('Distribution of Edge Distances')
ax7.axvline(x=3, color='red', linestyle='--', label='k=3 threshold')
ax7.axvline(x=4, color='orange', linestyle='--', label='r=4 threshold')
ax7.legend()

# Connectivity analysis
atom_degrees = np.bincount(all_edges[0][all_edges[0] < num_atoms])
grid_degrees = np.bincount(all_edges[1][all_edges[1] >= num_atoms] - num_atoms)

ax8.bar(['Atoms', 'Grid Points'], [atom_degrees.mean(), grid_degrees.mean()], 
        color=['blue', 'red'], alpha=0.7)
ax8.set_ylabel('Average Degree')
ax8.set_title('Average Node Connectivity')

plt.tight_layout()
plt.savefig('equi_grid_edge_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nConnectivity Statistics:")
print(f"- Average atom degree: {atom_degrees.mean():.2f}")
print(f"- Average grid point degree: {grid_degrees.mean():.2f}")
print(f"- Min/Max edge distance: {min(edge_distances):.2f} / {max(edge_distances):.2f}")