import argparse
import torch
from torch import nn, optim
import os
from torch_scatter import scatter_mean
import json
from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader
import numpy as np

# Import EnTransformer directly from en_transformer
from models.en_transformer.en_transformer import EnTransformer

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
        
        # Process each environment
        envs = []
        for i, (_, label_row) in enumerate(labels_df.iterrows()):
            env_indices = indices[i]
            if len(env_indices) > self.max_atoms:
                continue
                
            env_atoms = atoms_df.iloc[env_indices]
            
            # Get positions and atom type features
            positions = torch.FloatTensor(env_atoms[['x','y','z']].values)
            atom_features = torch.zeros((len(env_indices), len(self.atom_types)))
            for j, atype in enumerate(env_atoms['element']):
                if atype in self.atom_types:
                    atom_features[j, self.atom_types.index(atype)] = 1
                    
            # Get label
            label = self.aa_to_idx[label_row['label'] if isinstance(label_row['label'], str) 
                                 else self.amino_acids[label_row['label']]]
            
            envs.append((positions, atom_features, label))
            
        return envs

def collate_fn(batch):
    # Flatten batch of environments
    envs = [env for sample in batch for env in sample]
    if not envs:
        return None
    
    # Get dimensions
    n_envs = len(envs)
    max_atoms = max(env[0].size(0) for env in envs)
    
    # Initialize tensors
    pos = torch.zeros(n_envs, max_atoms, 3)
    feat = torch.zeros(n_envs, max_atoms, len(envs[0][1][0]))
    mask = torch.zeros(n_envs, max_atoms, dtype=torch.bool)
    labels = torch.LongTensor([env[2] for env in envs])
    
    # Fill tensors
    for i, (pos_i, feat_i, _) in enumerate(envs):
        n_atoms = pos_i.size(0)
        pos[i, :n_atoms] = pos_i
        feat[i, :n_atoms] = feat_i
        mask[i, :n_atoms] = 1
        
    return pos, feat, mask, labels

class EnTransformerResidueClassifier(nn.Module):
    def __init__(self, dim, depth, num_tokens=None, dim_head=64, heads=8, neighbors=0, checkpoint=False):
        super().__init__()
        
        self.transformer = EnTransformer(
            dim=dim,
            depth=depth,
            num_tokens=num_tokens,
            dim_head=dim_head,
            heads=heads,
            neighbors=neighbors,
            checkpoint=checkpoint
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 20)  # 20 amino acid classes
        )
        
    def forward(self, feat, coors, mask=None):
        # Process through transformer
        feat_out, _ = self.transformer(feat, coors, mask=mask)
        
        # Global pool and classify
        feat_pool = scatter_mean(feat_out, batch=torch.arange(coors.size(0), 
                               device=coors.device).repeat_interleave(coors.size(1)), dim=0)
        return self.mlp(feat_pool)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for pos, feat, mask, labels in loader:
        if pos is None:
            continue
            
        pos = pos.to(device)
        feat = feat.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(feat, pos, mask)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
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
        if pos is None:
            continue
            
        pos = pos.to(device)
        feat = feat.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        output = model(feat, pos, mask)
        loss = criterion(output, labels)
        
        total_loss += loss.item() * len(labels)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += len(labels)
        
    return total_loss / total, 100. * correct / total

def main(args):
    # Create datasets
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
    model = EnTransformerResidueClassifier(
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        neighbors=args.neighbors,
        checkpoint=args.checkpoint
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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
            
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
            
        with open(os.path.join(args.outf, args.exp_name, 'results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--neighbors', type=int, default=32)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--outf', type=str, default='res_outputs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(os.path.join(args.outf, args.exp_name), exist_ok=True)

    main(args)