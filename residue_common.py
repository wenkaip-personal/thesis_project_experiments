import torch
from torch import nn, optim
import os
import json
from torch_scatter import scatter_mean
from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader

class ResidueDataset:
    def __init__(self, lmdb_path, k_neighbors=16, radius=10.0, use_knn=True):
        self.dataset = LMDBDataset(lmdb_path)
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.use_knn = use_knn
        self.amino_acids = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
                           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP',
                           'TYR','VAL']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}

    def get_residue_features(self, atoms_df, residue_indices):
        # Calculate residue center (using CA atom or mean of all atoms)
        residue_pos = atoms_df.iloc[residue_indices][['x','y','z']].mean().values
        
        # Calculate residue features (you can modify this based on what features you want)
        residue_feats = torch.zeros(5)  # Example: 5 basic chemical properties
        # Add relevant residue features here
        
        return torch.tensor(residue_pos), residue_feats

    def build_graph(self, positions):
        positions = torch.tensor(positions)
        if self.use_knn:
            # Use KNN
            dists = torch.cdist(positions, positions)
            _, neighbor_idx = dists.topk(k=min(self.k_neighbors, len(positions)-1), 
                                       dim=-1, largest=False)
            neighbor_idx = neighbor_idx[:, 1:]  # Remove self-loops
        else:
            # Use radius graph
            dists = torch.cdist(positions, positions)
            neighbor_idx = (dists < self.radius).nonzero()
            # Remove self-loops
            neighbor_idx = neighbor_idx[neighbor_idx[:, 0] != neighbor_idx[:, 1]]
        
        return neighbor_idx

    def __getitem__(self, idx):
        data = self.dataset[idx]
        atoms_df = data['atoms']
        labels_df = data['labels']
        indices = data['subunit_indices']
        
        # Get residue positions and features
        positions = []
        features = []
        labels = []
        
        for i, (_, label_row) in enumerate(labels_df.iterrows()):
            pos, feat = self.get_residue_features(atoms_df, indices[i])
            positions.append(pos)
            features.append(feat)
            label = self.aa_to_idx[label_row['label'] if isinstance(label_row['label'], str) 
                                 else self.amino_acids[label_row['label']]]
            labels.append(label)

        positions = torch.stack(positions)
        features = torch.stack(features)
        labels = torch.tensor(labels)
        
        # Build graph connectivity
        edge_index = self.build_graph(positions)
        
        return positions, features, edge_index, labels

def collate_fn(batch):
    # Concatenate batch elements with offset for edge indices
    cumsum = 0
    all_pos = []
    all_feat = []
    all_edge_index = []
    all_labels = []
    
    for pos, feat, edge_index, labels in batch:
        all_pos.append(pos)
        all_feat.append(feat)
        all_edge_index.append(edge_index + cumsum)
        all_labels.append(labels)
        cumsum += len(pos)
    
    return (torch.cat(all_pos, dim=0),
            torch.cat(all_feat, dim=0),
            torch.cat(all_edge_index, dim=0),
            torch.cat(all_labels, dim=0))

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

def setup_training(args, model_class, model_kwargs):
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
    model = model_class(**model_kwargs).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion, train_loader, val_loader, test_loader

def train_model(args, model, optimizer, criterion, train_loader, val_loader, test_loader):
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