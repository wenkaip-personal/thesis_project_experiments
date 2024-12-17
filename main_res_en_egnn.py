import argparse
import torch
from torch import nn, optim
import os
import json
from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader
import numpy as np

# Import our EGNN model
from models.egnn.egnn_clean import EGNN

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='exp_1', help='Experiment name')
parser.add_argument('--batch_size', type=int, default=32)
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
            
            # Get positions
            positions = env_atoms[['x','y','z']].values
            
            # Create one-hot atom type features
            atom_types = env_atoms['element'].values
            unique_atoms = ['C', 'N', 'O', 'S', 'P']
            atom_features = np.zeros((len(atom_types), len(unique_atoms)))
            for j, atype in enumerate(atom_types):
                if atype in unique_atoms:
                    atom_features[j, unique_atoms.index(atype)] = 1
                    
            # Get label
            label = self.aa_to_idx[label_row['label']]
            
            pos_list.append(torch.FloatTensor(positions))
            feat_list.append(torch.FloatTensor(atom_features))
            label_list.append(label)
            
        return pos_list, feat_list, label_list

def collate_fn(batch):
    all_pos, all_feat, all_labels = [], [], []
    for pos, feat, labels in batch:
        all_pos.extend(pos)
        all_feat.extend(feat) 
        all_labels.extend(labels)
    return (
        torch.stack(all_pos),
        torch.stack(all_feat),
        torch.LongTensor(all_labels)
    )

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
        h, x = self.egnn(h, x, edges)
        # Global average pooling over nodes
        h = torch.mean(h, dim=0)
        return self.mlp(h)

model = EGNNForResidueIdentity(
    in_node_nf=5,  # Number of atom types
    hidden_nf=args.nf,
    out_node_nf=20,  # Number of amino acid classes
    n_layers=args.n_layers,
    attention=args.attention
).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (pos, feat, labels) in enumerate(train_loader):
        pos, feat, labels = pos.to(args.device), feat.to(args.device), labels.to(args.device)
        
        # Create edges - fully connected graph
        edges = torch.combinations(torch.arange(pos.size(0)), 2).t()
        edges = torch.cat([edges, edges.flip(0)], dim=1).to(args.device)
        
        optimizer.zero_grad()
        output = model(feat, pos, edges)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
    acc = 100. * correct / total
    return total_loss / len(train_loader), acc

def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pos, feat, labels in loader:
            pos, feat, labels = pos.to(args.device), feat.to(args.device), labels.to(args.device)
            edges = torch.combinations(torch.arange(pos.size(0)), 2).t()
            edges = torch.cat([edges, edges.flip(0)], dim=1).to(args.device)
            
            output = model(feat, pos, edges)
            loss = criterion(output, labels)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
    acc = 100. * correct / total
    return total_loss / len(loader), acc

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