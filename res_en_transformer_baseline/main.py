import argparse
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
import json
from dataset import RESDataset
from model import ResEnTransformer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='res_en_transformer_baseline')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_nf', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    return parser.parse_args()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Move data to device
        batch = batch.to(device)
        
        # Forward pass - only get predictions for central residues
        pred = model(batch.x, batch.pos, batch.edge_index, batch.central_indices)
        loss = criterion(pred, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch.y)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Move data to device
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch.x, batch.pos, batch.edge_index, batch.central_indices)
            loss = criterion(pred, batch.y)
            
            # Calculate accuracy
            pred_class = pred.argmax(dim=1)
            correct += (pred_class == batch.y).sum().item()
            total += batch.y.size(0)
            
            total_loss += loss.item() * len(batch.y)
            
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
    in_node_nf = sample.x.size(-1)  # Input feature dimension from node features
    out_node_nf = 20  # Number of amino acid classes
    
    # Initialize model
    model = ResEnTransformer(
        input_nf=in_node_nf,
        output_nf=out_node_nf,
        hidden_nf=args.hidden_nf,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
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
    
    # Save test results
    test_results = {'test_loss': test_loss, 'test_acc': test_acc}
    with open(f'results/{args.exp_name}_test_results.json', 'w') as f:
        json.dump(test_results, f)

if __name__ == '__main__':
    main()