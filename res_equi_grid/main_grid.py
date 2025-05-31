import argparse
import wandb
import torch
import torch.nn as nn
import tqdm
import time
import torch_geometric
from functools import partial
from atom3d.util import metrics
from dataset_grid import ProteinDataset
from model_grid import ProteinGrid
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='number of threads for loading data')
parser.add_argument('--batch', metavar='SIZE', type=int, default=8,
                    help='batch size')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time per training on trainset')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset')
parser.add_argument('--epochs', metavar='N', type=int, default=1000,
                    help='training epochs')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--hidden_nf', type=int, default=128,
                    help='number of hidden features')
parser.add_argument('--grid-size', type=int, default=9,
                    help='size of the grid for gridification')
parser.add_argument('--grid-spacing', type=int, default=8, 
                    help='spacing of the grid points')
parser.add_argument('--data_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/')
parser.add_argument('--split_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/')
parser.add_argument('--model_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/thesis_project_experiments/res_equi_grid/models/')
parser.add_argument('--debug', action='store_true',
                    help='enable debug mode with reduced dataset size and fewer epochs')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = int(time.time())
print = partial(print, flush=True)

# Initialize wandb
run = wandb.init(
    project="res",
    config={
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "architecture": "ResEquiGrid",
        "dataset": "RES",
        "epochs": args.epochs,
        "grid_size": args.grid_size,
        "grid_spacing": args.grid_spacing,
        "batch_size": args.batch,
    },
)

def loop(dataloader, model, optimizer=None, scheduler=None, max_time=None, max_batches=None):
    start = time.time()
    t = tqdm.tqdm(dataloader)
    total_loss, total_count = 0, 0
    all_acc = []
    all_losses = []

    batch_count = 0
    for batch in t:
        if max_batches is not None and batch_count >= max_batches:
            break
        
        if max_time and (time.time() - start) > 60*max_time: 
            break
            
        if optimizer:
            optimizer.zero_grad()
            
        try:
            loss, log_dict = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        
        batch_loss = float(loss)
        total_loss += batch_loss
        total_count += 1
        all_losses.append(batch_loss)
        
        # Store batch accuracy
        batch_acc = log_dict['acc'].mean().item()
        all_acc.append(batch_acc)

        if optimizer:
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
        
        batch_count += 1        
        t.set_description(f"Loss: {total_loss/total_count:.4f}, Acc: {batch_acc:.1f}%")
        
        # Log metrics to wandb
        run.log({
            "loss": total_loss / total_count,
            "accuracy": batch_acc
        })
        
    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    avg_acc = sum(all_acc) / len(all_acc) if all_acc else 0
    std_acc = np.std(all_acc) if len(all_acc) > 1 else 0
    
    return avg_loss, avg_acc, std_acc

def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    patience = 50
    patience_counter = 0

    # Determine max batches for debug mode
    train_batches = 10 if args.debug else None
    val_batches = 5 if args.debug else None

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_acc, train_std = loop(train_loader, model, optimizer=optimizer, 
                                               scheduler=scheduler, max_time=args.train_time, 
                                               max_batches=train_batches)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_std = loop(val_loader, model, max_time=args.val_time, 
                                             max_batches=val_batches)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        
        path = os.path.join(args.model_path, f'RES_{model_id}_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = path
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'\nEPOCH {epoch}:')
        print(f'  TRAIN - Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% (±{train_std:.1f}%)')
        print(f'  VAL   - Loss: {val_loss:.4f}, Acc: {val_acc:.1f}% (±{val_std:.1f}%)')
        print(f'  BEST  - Epoch: {best_epoch}, Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.1f}%')

        # Log epoch metrics
        run.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

def test(model, test_loader, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_batches = 5 if args.debug else None
    
    with torch.no_grad():
        test_loss, test_acc, test_std = loop(test_loader, model, max_batches=test_batches)
    
    print(f'\nTEST Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  Acc:  {test_acc:.1f}% (±{test_std:.1f}%)')

    run.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })

def forward(model, batch, device):
    batch = batch.to(device)
    return model(batch)

def main():
    # Create model directory if it doesn't exist
    os.makedirs(args.model_path, exist_ok=True)
    
    # Adjust parameters for debug mode
    if args.debug:
        print("Running in DEBUG mode")
        max_samples = 100
        args.epochs = 50
    else:
        max_samples = None

    # Load datasets
    dataset = ProteinDataset(
        lmdb_path=args.data_path,
        split_path_root=args.split_path,
        batch_size=args.batch,
        size=args.grid_size,
        spacing=args.grid_spacing,
        max_samples=max_samples
    )

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    # Initialize model
    model = ProteinGrid(
        node_types=9,
        res_types=21,
        on_bb=2,
        hidden_features=args.hidden_nf,
        out_features=20,
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")

    if args.test:
        test(model, test_loader, args.test)
    else:
        train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
    run.finish()