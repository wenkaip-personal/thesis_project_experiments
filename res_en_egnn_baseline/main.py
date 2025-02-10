import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import json
from dataset import RESDataset
from model import ResEGNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='res_egnn_baseline')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_nf', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    return parser.parse_args()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for h, x, edges, y in loader:
        optimizer.zero_grad()
        
        # Move data to device
        h = h.to(device)
        x = x.to(device)
        edges = edges.to(device)
        y = y.to(device)
        
        # Forward pass
        pred = model(h, x, edges)
        loss = criterion(pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * loader.batch_size
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for h, x, edges, y in loader:
            # Move data to device
            h = h.to(device)
            x = x.to(device)
            edges = edges.to(device)
            y = y.to(device)
            
            # Forward pass
            pred = model(h, x, edges)
            loss = criterion(pred, y)
            
            # Calculate accuracy
            pred_class = pred.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)
            
            total_loss += loss.item() * loader.batch_size
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, accuracy

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    base_path = '/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data'
    train_dataset = RESDataset(f'{base_path}/train')
    val_dataset = RESDataset(f'{base_path}/val')
    test_dataset = RESDataset(f'{base_path}/test')
    
    # If debug mode, use small subset
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(50))
        test_dataset = torch.utils.data.Subset(test_dataset, range(50))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get input/output dimensions from data
    sample = next(iter(train_loader))
    in_node_nf = sample[0].size(-1)  # Input feature dimension
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
            torch.save(model.state_dict(), f'models/{args.exp_name}_best.pt')
        
        # Save results
        with open(f'results/{args.exp_name}_results.json', 'w') as f:
            json.dump(results, f)
    
    # Test best model
    model.load_state_dict(torch.load(f'models/{args.exp_name}_best.pt'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()