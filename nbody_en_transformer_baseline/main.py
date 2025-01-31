import argparse
import torch
from torch import nn, optim
import json
from dataset import NBodyTransformerDataset
from model import NBodyTransformer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_nf', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--max_training_samples', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default="nbody_small")
    return parser.parse_args()

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (loc, vel, charges, loc_end) in enumerate(loader):
        optimizer.zero_grad()
        
        # Move data to device
        loc = loc.to(device)
        vel = vel.to(device)
        charges = charges.to(device)
        loc_end = loc_end.to(device)
        
        # Forward pass
        pred_pos = model(charges, loc, vel)
        loss = nn.MSELoss()(pred_pos, loc_end)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * loader.batch_size
        
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for loc, vel, charges, loc_end in loader:
            # Move data to device
            loc = loc.to(device)
            vel = vel.to(device) 
            charges = charges.to(device)
            loc_end = loc_end.to(device)
            
            # Forward pass
            pred_pos = model(charges, loc, vel)
            loss = nn.MSELoss()(pred_pos, loc_end)
            
            total_loss += loss.item() * loader.batch_size
            
    return total_loss / len(loader.dataset)

def test_final(model, test_loader, device):
    args = get_args()

    # Load best model
    model.load_state_dict(torch.load(f'nbody_en_transformer_baseline/models/{args.exp_name}_best.pt'))
    model.eval()
    
    total_loss = 0.0
    total_forward_time = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for loc, vel, charges, loc_end in test_loader:
            # Move data to device
            loc = loc.to(device)
            vel = vel.to(device)
            charges = charges.to(device) 
            loc_end = loc_end.to(device)
            
            # Time the forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            pred_pos = model(charges, loc, vel)
            end_time.record()
            
            # Wait for GPU completion
            torch.cuda.synchronize()
            
            # Calculate loss and timing
            loss = nn.MSELoss()(pred_pos, loc_end)
            forward_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            total_loss += loss.item() * test_loader.batch_size
            total_forward_time += forward_time
            n_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / len(test_loader.dataset)
    avg_forward_time = total_forward_time / n_batches
    
    print(f'\nFinal Test Results:')
    print(f'Mean Squared Error: {avg_loss:.4f}')
    print(f'Average Forward Time: {avg_forward_time:.4f} seconds')
    
    # Save final results
    final_results = {
        'test_mse': avg_loss,
        'forward_time': avg_forward_time
    }
    with open(f'nbody_en_transformer_baseline/results/{args.exp_name}_final_results.json', 'w') as f:
        json.dump(final_results, f)

def main():
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load datasets
    train_dataset = NBodyTransformerDataset('train', args.max_training_samples, args.dataset)
    val_dataset = NBodyTransformerDataset('val', args.max_training_samples, "nbody_small") 
    test_dataset = NBodyTransformerDataset('test', args.max_training_samples, "nbody_small")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = NBodyTransformer(
        hidden_nf=args.hidden_nf,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        device=device
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    results = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'nbody_en_transformer_baseline/models/{args.exp_name}_best.pt')
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save results
        with open(f'nbody_en_transformer_baseline/results/{args.exp_name}_results.json', 'w') as f:
            json.dump(results, f)

    test_final(model, test_loader, device)

if __name__ == '__main__':
    main()