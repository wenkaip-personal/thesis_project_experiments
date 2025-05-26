import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset_grid import Protein, _element_mapping, _amino_acids
from torch_cluster import knn, radius
import matplotlib.patches as mpatches
import os

# Create output directory for plots
output_dir = '/content/drive/MyDrive/thesis_project/thesis_project_experiments/res_equi_grid/visualizations/'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/'
split_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/train_indices.txt'

# Initialize the dataset with visualization-friendly parameters
dataset = Protein(
    lmdb_path=data_path,
    split_path=split_path,
    radius=4.5,
    k=3,
    knn=True,
    size=5,  # Using smaller grid for clearer visualization
    spacing=4,
    max_samples=10
)

# Get a sample
sample_idx = 0
grid_data = dataset[sample_idx]

# Extract data from the sample
atom_coords = grid_data.coords.numpy()
grid_coords = grid_data.grid_coords.numpy()
atom_types = grid_data.atom_types.numpy()
ca_idx = grid_data.cb_index.item()  # Central amino acid index
target_aa = grid_data.y.item()  # Target amino acid type

# Get atom type names for labeling
element_names = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Other']
atom_elements = [element_names[at] for at in atom_types]

# Extract edge information
edge_index = grid_data.grid_edge_index.numpy()
edges_atom_to_grid = []
edges_grid_to_atom = []

for i in range(edge_index.shape[1]):
    src, dst = edge_index[0, i], edge_index[1, i]
    if src < len(atom_coords) and dst >= len(atom_coords):
        # Atom to grid edge
        edges_atom_to_grid.append((src, dst - len(atom_coords)))
    elif src >= len(atom_coords) and dst < len(atom_coords):
        # Grid to atom edge
        edges_grid_to_atom.append((src - len(atom_coords), dst))

# Create the visualization
fig = plt.figure(figsize=(16, 6))

# First subplot: Atom positions only
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('Protein Residue Environment\n(Centered at CA atom)', fontsize=12)

# Plot atoms with different colors for different elements
colors = ['red', 'gray', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown']
for i, (coord, elem) in enumerate(zip(atom_coords, atom_types)):
    if i == ca_idx:
        ax1.scatter(coord[0], coord[1], coord[2], c='gold', s=200, marker='*', 
                   edgecolors='black', linewidth=2, label='Central CA atom')
    else:
        ax1.scatter(coord[0], coord[1], coord[2], c=colors[elem], s=50, alpha=0.8)

ax1.set_xlabel('X (Å)')
ax1.set_ylabel('Y (Å)')
ax1.set_zlabel('Z (Å)')
ax1.legend()

# Second subplot: Grid points only
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title(f'3D Grid Structure\n({dataset.size}×{dataset.size}×{dataset.size} points)', 
              fontsize=12)

# Plot grid points
ax2.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='lightblue', s=30, alpha=0.6, marker='s')

ax2.set_xlabel('X (Å)')
ax2.set_ylabel('Y (Å)')
ax2.set_zlabel('Z (Å)')

# Third subplot: Combined view with edges
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title('Atom-Grid Connectivity\n(k-NN and radius edges)', fontsize=12)

# Plot atoms
for i, (coord, elem) in enumerate(zip(atom_coords, atom_types)):
    if i == ca_idx:
        ax3.scatter(coord[0], coord[1], coord[2], c='gold', s=200, marker='*', 
                   edgecolors='black', linewidth=2)
    else:
        ax3.scatter(coord[0], coord[1], coord[2], c=colors[elem], s=50, alpha=0.8)

# Plot grid points
ax3.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 
           c='lightblue', s=20, alpha=0.4, marker='s')

# Plot edges
edges_to_plot = edge_index

for atom_idx, grid_idx in edges_to_plot:
    atom_pos = atom_coords[atom_idx]
    grid_pos = grid_coords[grid_idx]
    ax3.plot([atom_pos[0], grid_pos[0]], 
            [atom_pos[1], grid_pos[1]], 
            [atom_pos[2], grid_pos[2]], 
            'gray', alpha=0.3, linewidth=0.5)

ax3.set_xlabel('X (Å)')
ax3.set_ylabel('Y (Å)')
ax3.set_zlabel('Z (Å)')

# Add title with dataset information
aa_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
target_aa_name = aa_names[target_aa] if target_aa < 20 else 'Unknown'

plt.tight_layout()

# Save the 3D visualization
fig.savefig(os.path.join(output_dir, 'atom3d_res_3d_visualization.png'), 
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'atom3d_res_3d_visualization.pdf'), 
            bbox_inches='tight')
plt.show()

# Create a 2D projection for clearer edge visualization
fig2, ax = plt.subplots(figsize=(10, 8))
ax.set_title('2D Projection (XY plane) - Atom-Grid Connectivity', fontsize=14)

# Plot atoms
for i, (coord, elem) in enumerate(zip(atom_coords, atom_types)):
    if i == ca_idx:
        ax.scatter(coord[0], coord[1], c='gold', s=300, marker='*', 
                  edgecolors='black', linewidth=2, zorder=5)
        ax.annotate('CA', (coord[0], coord[1]), xytext=(5, 5), 
                   textcoords='offset points', fontweight='bold')
    else:
        ax.scatter(coord[0], coord[1], c=colors[elem], s=100, alpha=0.8, zorder=3)

# Plot grid points
ax.scatter(grid_coords[:, 0], grid_coords[:, 1], c='lightblue', s=50, 
          alpha=0.5, marker='s', zorder=2)

# Plot edges
for atom_idx, grid_idx in edges_to_plot:
    atom_pos = atom_coords[atom_idx]
    grid_pos = grid_coords[grid_idx]
    ax.plot([atom_pos[0], grid_pos[0]], 
           [atom_pos[1], grid_pos[1]], 
           'gray', alpha=0.2, linewidth=0.8, zorder=1)

ax.set_xlabel('X (Å)', fontsize=12)
ax.set_ylabel('Y (Å)', fontsize=12)
ax.grid(True, alpha=0.3)

# Create legend for atom types
legend_elements = []
for i, elem_name in enumerate(element_names[:8]):
    if i in atom_types:
        legend_elements.append(mpatches.Patch(color=colors[i], label=elem_name))
legend_elements.append(mpatches.Patch(color='gold', label='Central CA'))
legend_elements.append(mpatches.Patch(color='lightblue', label='Grid points'))

ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()

# Save the 2D projection
fig2.savefig(os.path.join(output_dir, 'atom3d_res_2d_projection.png'), 
             dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'atom3d_res_2d_projection.pdf'), 
             bbox_inches='tight')
plt.show()

# Create individual component plots for detailed analysis
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

# Atom distribution histogram
ax = axes[0, 0]
unique_atoms, counts = np.unique(atom_types, return_counts=True)
atom_names = [element_names[i] for i in unique_atoms]
ax.bar(atom_names, counts, color=[colors[i] for i in unique_atoms])
ax.set_xlabel('Atom Type')
ax.set_ylabel('Count')
ax.set_title('Atom Type Distribution')

# Distance distribution
ax = axes[0, 1]
distances_from_ca = np.linalg.norm(atom_coords - atom_coords[ca_idx], axis=1)
ax.hist(distances_from_ca[distances_from_ca > 0], bins=20, alpha=0.7, color='steelblue')
ax.set_xlabel('Distance from CA (Å)')
ax.set_ylabel('Count')
ax.set_title('Distance Distribution from Central CA')

# Edge degree distribution
ax = axes[1, 0]
atom_degrees = np.bincount(edge_index[0][edge_index[0] < len(atom_coords)])
ax.hist(atom_degrees, bins=np.arange(0, atom_degrees.max() + 2) - 0.5, 
        alpha=0.7, color='darkgreen')
ax.set_xlabel('Degree (number of connections)')
ax.set_ylabel('Count')
ax.set_title('Atom Connectivity Degree Distribution')

# Grid utilization
ax = axes[1, 1]
grid_degrees = np.bincount(edge_index[1][edge_index[1] >= len(atom_coords)] - len(atom_coords))
utilized_grid_points = np.sum(grid_degrees > 0)
total_grid_points = len(grid_coords)
labels = ['Connected', 'Unconnected']
sizes = [utilized_grid_points, total_grid_points - utilized_grid_points]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
       colors=['lightgreen', 'lightcoral'])
ax.set_title(f'Grid Point Utilization\n({utilized_grid_points}/{total_grid_points} connected)')

plt.suptitle(f'ATOM3D RES Sample Analysis - Target: {target_aa_name}', fontsize=14)
plt.tight_layout()

# Save the analysis plots
fig3.savefig(os.path.join(output_dir, 'atom3d_res_sample_analysis.png'), 
             dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(output_dir, 'atom3d_res_sample_analysis.pdf'), 
             bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\nDataset Sample Summary:")
print(f"- Target amino acid: {target_aa_name} (class {target_aa})")
print(f"- Number of atoms: {len(atom_coords)}")
print(f"- Number of grid points: {len(grid_coords)}")
print(f"- Total edges: {edge_index.shape[1]}")
print(f"- Central CA atom index: {ca_idx}")
print(f"- Coordinate range: X [{atom_coords[:, 0].min():.2f}, {atom_coords[:, 0].max():.2f}], "
      f"Y [{atom_coords[:, 1].min():.2f}, {atom_coords[:, 1].max():.2f}], "
      f"Z [{atom_coords[:, 2].min():.2f}, {atom_coords[:, 2].max():.2f}]")
print(f"- Grid dimensions: {dataset.size}×{dataset.size}×{dataset.size}")
print(f"- Edge search radius: {dataset.radius} Å")
print(f"- k-nearest neighbors: {dataset.k}")
print(f"\nPlots saved to: {output_dir}")