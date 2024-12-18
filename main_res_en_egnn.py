import argparse
import torch
from torch import nn, optim
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json
from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader
import numpy as np

# Import our EGNN model
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
    def __init__(self, lmdb_path):
        self.dataset = LMDBDataset(lmdb_path)
        self.amino_acids = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
                           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP',
                           'TYR','VAL']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Get atom positions and features
        atoms_df = data['atoms']
        labels_df = data['labels']
        indices = data['subunit_indices']
        
        # Process each residue environment
        pos_list = []
        feat_list = []
        label_list = []
        
        for i, (_, label_row) in enumerate(labels_df.iterrows()):
            env_indices = indices[i]
            env_atoms = atoms_df.iloc[env_indices]

            # Skip environments that are too large
            if len(env_atoms) > 1000:  # Adjust threshold as needed
                continue
            
            # Get positions
            positions = env_atoms[['x','y','z']].values
            
            # Create one-hot atom type features
            atom_types = env_atoms['element'].values
            unique_atoms = ['C', 'N', 'O', 'S', 'P']
            atom_features = np.zeros((len(atom_types), len(unique_atoms)))
            for j, atype in enumerate(atom_types):
                if atype in unique_atoms:
                    atom_features[j, unique_atoms.index(atype)] = 1
                    
            # Get label - handle potential numeric labels
            label = label_row['label']
            if isinstance(label, (int, np.integer)):
                label = self.amino_acids[label]
            label_idx = self.aa_to_idx[label]
            
            pos_list.append(torch.FloatTensor(positions))
            feat_list.append(torch.FloatTensor(atom_features))
            label_list.append(label_idx)
            
        return pos_list, feat_list, label_list

def collate_fn(batch):
    # Instead of stacking tensors directly, we'll return them as lists
    # Each item in these lists corresponds to one residue environment
    all_pos, all_feat, all_labels = [], [], []
    
    # Unpack the nested batch structure 
    for sample in batch:
        # Each sample contains lists of pos, feat, labels
        pos_list, feat_list, labels_list = sample
        
        # Extend our lists with data for each residue environment
        all_pos.extend([pos.to(torch.float32) for pos in pos_list])
        all_feat.extend([feat.to(torch.float32) for feat in feat_list])
        all_labels.extend(labels_list)
        
    # Return lists rather than stacked tensors
    return all_pos, all_feat, torch.LongTensor(all_labels)

# Initialize datasets and dataloaders
train_dataset = ResidueDataset(os.path.join(args.dataset_path, 'train'))
val_dataset = ResidueDataset(os.path.join(args.dataset_path, 'val'))
test_dataset = ResidueDataset(os.path.join(args.dataset_path, 'test'))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                       shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)

# Initialize model
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
        
    def forward(self, h, x, edges):
        # Process single environment
        # Add edge attributes (None since we don't have any special edge features)
        edge_attr = None
        h, x = self.egnn(h, x, edges, edge_attr)  # Add edge_attr argument
        # Global average pooling over nodes 
        h = torch.mean(h, dim=0)
        # Return logits for amino acid prediction
        return self.mlp(h).unsqueeze(0)  # Add batch dimension

model = EGNNForResidueIdentity(
    in_node_nf=5,  # Number of atom types
    hidden_nf=args.nf,
    out_node_nf=20,  # Number of amino acid classes
    n_layers=args.n_layers,
    attention=args.attention
).to(args.device)

def create_edges_with_radius_cutoff(pos, cutoff=10.0):
    # Ensure pos is on CPU for the initial computations
    pos_cpu = pos.cpu()
    n_atoms = pos_cpu.size(0)
    
    # Get all pairs of indices
    rows, cols = torch.combinations(torch.arange(n_atoms), 2).t()
    
    # Calculate distances using CPU tensors
    distances = torch.norm(pos_cpu[rows] - pos_cpu[cols], dim=1)
    
    # Create mask and filter
    mask = distances < cutoff
    rows = rows[mask]
    cols = cols[mask]
    
    # Add reverse edges
    edges = torch.cat([
        torch.stack([rows, cols]),
        torch.stack([cols, rows])
    ], dim=1)
    
    # Move final result to same device as input pos
    return edges.to(pos.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (pos_list, feat_list, labels) in enumerate(train_loader):
        # Process each chunk of environments
        chunk_size = 4  # Process 4 environments at a time
        for i in range(0, len(pos_list), chunk_size):
            # Get chunk of data
            chunk_pos = pos_list[i:i+chunk_size]
            chunk_feat = feat_list[i:i+chunk_size] 
            chunk_labels = labels[i:i+chunk_size].to(args.device)
            
            # Initialize outputs list for this chunk
            chunk_outputs = []
            
            # Process each environment in the chunk
            for pos, feat in zip(chunk_pos, chunk_feat):
                pos = pos.to(args.device)
                feat = feat.to(args.device)
                
                # Create edges for this environment
                edges = create_edges_with_radius_cutoff(pos).to(args.device)
                
                # Get model output for this environment
                output = model(feat, pos, edges)
                chunk_outputs.append(output)
                
                # Clear some memory
                torch.cuda.empty_cache()
            
            # Combine outputs from this chunk
            chunk_output = torch.cat(chunk_outputs, dim=0)
            
            # Calculate loss for this chunk
            optimizer.zero_grad()
            loss = criterion(chunk_output, chunk_labels)
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = chunk_output.argmax(dim=1)
            correct += pred.eq(chunk_labels).sum().item()
            total += chunk_labels.size(0)
            
            # Clear more memory
            torch.cuda.empty_cache()
            
        # Print progress 
        if batch_idx % 10 == 0:
            current = batch_idx * args.batch_size
            print(f'Train Epoch: {epoch} [{current}/{len(train_loader.dataset)} '
                  f'({100. * current / len(train_loader.dataset):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    # Calculate final metrics
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total
    
    return avg_loss, acc

def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pos_list, feat_list, labels in loader:
            # Process in chunks
            chunk_size = 4
            for i in range(0, len(pos_list), chunk_size):
                # Get chunk of data
                chunk_pos = pos_list[i:i+chunk_size]
                chunk_feat = feat_list[i:i+chunk_size]
                chunk_labels = labels[i:i+chunk_size].to(args.device)
                
                chunk_outputs = []
                
                # Process each environment in chunk
                for pos, feat in zip(chunk_pos, chunk_feat):
                    pos = pos.to(args.device)
                    feat = feat.to(args.device)
                    
                    edges = create_edges_with_radius_cutoff(pos).to(args.device)
                    output = model(feat, pos, edges)
                    chunk_outputs.append(output)
                    
                    torch.cuda.empty_cache()
                
                # Combine chunk outputs
                chunk_output = torch.cat(chunk_outputs, dim=0)
                
                # Calculate loss
                loss = criterion(chunk_output, chunk_labels)
                
                # Update metrics
                total_loss += loss.item()
                pred = chunk_output.argmax(dim=1)
                correct += pred.eq(chunk_labels).sum().item()
                total += chunk_labels.size(0)
                
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

# Main training loop
best_val_acc = 0
results = {'train_loss': [], 'train_acc': [], 
           'val_loss': [], 'val_acc': [],
           'test_loss': [], 'test_acc': []}

for epoch in range(args.epochs):
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['val_loss'].append(val_loss)
    results['val_acc'].append(val_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)
    
    print(f'Epoch {epoch}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                  os.path.join(args.outf, args.exp_name, 'best_model.pt'))
        
    # Save results
    with open(os.path.join(args.outf, args.exp_name, 'results.json'), 'w') as f:
        json.dump(results, f)