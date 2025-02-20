import argparse
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
import json
from dataset import RESDataset
from model import ResEGNN
from functools import partial

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='res_egnn_baseline')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_nf', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/')
    parser.add_argument('--split_path', type=str, default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/')
    return parser.parse_args()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Convert atoms tensor to float and ensure it requires gradients
        atoms_float = batch.atoms.float().requires_grad_()
        x = batch.x.requires_grad_()
        
        pred = model(atoms_float, x, batch.edge_index, batch)
        loss = criterion(pred, batch.label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.label.size(0)
        n_samples += batch.label.size(0)
        
    return total_loss / n_samples

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    correct = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Convert atoms tensor to float before passing to model
            atoms_float = batch.atoms.float()
            
            pred = model(atoms_float, batch.x, batch.edge_index, batch)
            loss = criterion(pred, batch.label)
            
            pred_class = pred.argmax(dim=1)
            correct += (pred_class == batch.label).sum().item()
            
            total_loss += loss.item() * batch.label.size(0)
            n_samples += batch.label.size(0)
            
    accuracy = 100 * correct / n_samples
    avg_loss = total_loss / n_samples
    
    return avg_loss, accuracy

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    data_path = args.data_path
    split_path = args.split_path
    dataset = partial(RESDataset, data_path) 
    train_dataset = dataset(split_path=split_path + 'train_indices.txt')
    val_dataset = dataset(split_path=split_path + 'val_indices.txt')
    test_dataset = dataset(split_path=split_path + 'test_indices.txt')
    
    # Create dataloaders - note we don't shuffle since it's an IterableDataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get input dimensions from data
    in_node_nf = 1  # Since we're using one-hot encoded atoms
    out_node_nf = 20  # Number of amino acid classes
    
    # Initialize model
    model = ResEGNN(
        in_node_nf=in_node_nf,
        hidden_nf=args.hidden_nf,
        out_node_nf=out_node_nf,
        n_layers=args.n_layers,
        device=device
    )
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    results = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'res_en_egnn_baseline/models/{args.exp_name}_best.pt')
        
        # Save results
        with open(f'res_en_egnn_baseline/results/{args.exp_name}_results.json', 'w') as f:
            json.dump(results, f)
    
    # Test best model
    model.load_state_dict(torch.load(f'res_en_egnn_baseline/models/{args.exp_name}_best.pt'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    # Save test results
    test_results = {'test_loss': test_loss, 'test_acc': test_acc}
    with open(f'res_en_egnn_baseline/results/{args.exp_name}_test_results.json', 'w') as f:
        json.dump(test_results, f)

if __name__ == '__main__':
    main()