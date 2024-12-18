import argparse
import torch
from torch import nn, optim
import os
from torch_scatter import scatter_mean
import json
from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader
import numpy as np

# Import EGNN model
from models.egnn.egnn_clean import EGNN

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='exp_1', help='Experiment name')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--nf', type=int, default=128, help='Number of features')
parser.add_argument('--n_layers', type=int, default=7)
parser.add_argument('--attention', type=int, default=1)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--outf', type=str, default='res_outputs')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# Create output directory
os.makedirs(args.outf, exist_ok=True)
os.makedirs(os.path.join(args.outf, args.exp_name), exist_ok=True)

class ResidueDataset:
    def __init__(self, lmdb_path, max_atoms=1000):
        self.dataset = LMDBDataset(lmdb_path)
        self.max_atoms = max_atoms
        self.amino_acids = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
                           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP',
                           'TYR','VAL']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.atom_types = ['C', 'N', 'O', 'S', 'P']
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        atoms_df = data['atoms']
        labels_df = data['labels']
        indices = data['subunit_indices']
        
        pos_list = []
        feat_list = []
        label_list = []
        
        for i, (_, label_row) in enumerate(labels_df.iterrows()):
            env_indices = indices[i]
            if len(env_indices) > self.max_atoms:
                continue
                
            env_atoms = atoms_df.iloc[env_indices]
            
            # Get positions
            positions = env_atoms[['x','y','z']].values
            
            # Create atom type features
            atom_types = env_atoms['element'].values
            atom_features = np.zeros((len(atom_types), len(self.atom_types)))
            for j, atype in enumerate(atom_types):
                if atype in self.atom_types:
                    atom_features[j, self.atom_types.index(atype)] = 1
            
            # Get label
            label = label_row['label']
            if isinstance(label, (int, np.integer)):
                label = self.amino_acids[label]
            label_idx = self.aa_to_idx[label]
            
            pos_list.append(torch.FloatTensor(positions))
            feat_list.append(torch.FloatTensor(atom_features))
            label_list.append(label_idx)
            
        return pos_list, feat_list, label_list

def collate_fn(batch):
    # Get total number of environments and max atoms
    n_envs = sum(len(sample[0]) for sample in batch)
    max_atoms = max(pos.size(0) for sample in batch for pos in sample[0])
    
    # Pre-allocate tensors
    pos_tensor = torch.zeros(n_envs, max_atoms, 3)
    feat_tensor = torch.zeros(n_envs, max_atoms, 5)
    mask_tensor = torch.zeros(n_envs, max_atoms, dtype=torch.bool)
    labels = []
    
    # Fill tensors
    idx = 0
    for sample in batch:
        pos_list, feat_list, label_list = sample
        for pos, feat, label in zip(pos_list, feat_list, label_list):
            n_atoms = pos.size(0)
            pos_tensor[idx, :n_atoms] = pos
            feat_tensor[idx, :n_atoms] = feat
            mask_tensor[idx, :n_atoms] = 1
            labels.append(label)
            idx += 1
            
    return pos_tensor, feat_tensor, mask_tensor, torch.LongTensor(labels)

class EGNNForResidueIdentity(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers, attention):
        super().__init__()
        self.egnn = EGNN(in_node_nf=in_node_nf,
                        hidden_nf=hidden_nf,
                        out_node_nf=hidden_nf,
                        n_layers=n_layers,
                        attention=attention)
                        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, out_node_nf)
        )
        
    def forward(self, h, x, edges, batch):
        # Process through EGNN
        h, x = self.egnn(h, x, edges, edge_attr=None)
        
        # Global pool per environment using batch indices
        h_pool = scatter_mean(h, batch, dim=0)
        
        # Final classification
        return self.mlp(h_pool)

def create_edges_batch(pos_tensor, mask_tensor, cutoff=10.0):
    """
    Creates edges for batch of environments efficiently
    Args:
        pos_tensor: [B, N, 3] tensor of positions
        mask_tensor: [B, N] boolean mask of valid atoms 
        cutoff: Distance cutoff for edges
    Returns:
        edges: [2, E] tensor of edges
        batch: [E] tensor assigning edges to environments
    """
    B, N = pos_tensor.shape[:2]
    device = pos_tensor.device
    
    # Create indices for all pairs within cutoff
    rows, cols = torch.triu_indices(N, N, 1, device=device)
    rows = rows.repeat(B)
    cols = cols.repeat(B)
    batch = torch.arange(B, device=device).repeat_interleave(rows.size(0)//B)
    
    # Calculate pairwise distances
    distances = torch.norm(
        pos_tensor[batch, rows] - pos_tensor[batch, cols],
        dim=-1
    )
    
    # Filter edges by cutoff and mask
    valid = (distances < cutoff) & \
            mask_tensor[batch, rows] & \
            mask_tensor[batch, cols]
            
    rows = rows[valid]
    cols = cols[valid]
    batch = batch[valid]
    
    # Create bidirectional edges
    edges = torch.stack([
        torch.cat([rows, cols]),
        torch.cat([cols, rows])
    ])
    batch = torch.cat([batch, batch])
    
    return edges, batch

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for pos, feat, mask, labels in loader:
        # Move to device
        pos = pos.to(device)
        feat = feat.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        # Create edges
        edges, batch = create_edges_batch(pos, mask)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(feat, pos, edges, batch)
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * len(labels)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += len(labels)
        
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for pos, feat, mask, labels in loader:
        # Move to device
        pos = pos.to(device)
        feat = feat.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        # Create edges
        edges, batch = create_edges_batch(pos, mask)
        
        # Forward pass
        output = model(feat, pos, edges, batch)
        loss = criterion(output, labels)
        
        # Update metrics
        total_loss += loss.item() * len(labels)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += len(labels)
        
    return total_loss / total, 100. * correct / total

def main():
    # Initialize datasets
    train_dataset = ResidueDataset(os.path.join(args.dataset_path, 'train'))
    val_dataset = ResidueDataset(os.path.join(args.dataset_path, 'val'))
    test_dataset = ResidueDataset(os.path.join(args.dataset_path, 'test'))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Initialize model
    model = EGNNForResidueIdentity(
        in_node_nf=5,  # Number of atom types
        hidden_nf=args.nf,
        out_node_nf=20,  # Number of amino acid classes
        n_layers=args.n_layers,
        attention=args.attention
    ).to(args.device)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device)
            
        # Evaluate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, args.device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, args.device)
            
        # Save results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
        # Print progress
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                      os.path.join(args.outf, args.exp_name, 'best_model.pt'))
            
        # Save results
        with open(os.path.join(args.outf, args.exp_name, 'results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()