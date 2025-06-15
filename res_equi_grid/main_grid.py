import argparse
import wandb
import torch
import torch.nn as nn
import tqdm
import time
# import torch_geometric
from functools import partial
from atom3d.util import metrics
from dataset_grid import ProteinDataset
from model_grid import ProteinGrid
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='number of threads for loading data')
parser.add_argument('--batch', metavar='SIZE', type=int, default=1024,
                    help='batch size')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time per training on trainset')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--hidden_nf', type=int, default=128,
                    help='number of hidden features')
parser.add_argument('--grid-size', type=int, default=5,
                    help='size of the grid for gridification')
parser.add_argument('--grid-spacing', type=int, default=4, 
                    help='spacing of the grid points')
parser.add_argument('--data_path', type=str, 
                    default='./raw/RES/data/')
parser.add_argument('--split_path', type=str, 
                    default='./indices/')
parser.add_argument('--model_path', type=str, 
                    default='./thesis_project_experiments/res_equi_grid/models/')
# Add debug mode flag
parser.add_argument('--debug', action='store_true',
                    help='enable debug mode with reduced dataset size and fewer epochs')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())
print = partial(print, flush=True)

# Start a new wandb run to track this script
run = wandb.init(
    # Set the wandb project where this run will be logged
    project="res",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": "ResEquiGrid",
        "dataset": "RES",
        "epochs": args.epochs,
        "grid_size": args.grid_size,
        "grid_spacing": args.grid_spacing,
    },
)

def loop(dataloader, model, optimizer=None, max_time=None, max_batches=None):
    start = time.time()
    t = tqdm.tqdm(dataloader)
    metrics_funcs = get_metrics()
    total_loss, total_count = 0, 0
    all_acc = []  # Store accuracy values

    batch_count = 0
    for batch in t:
        # Add max_batches check for debug mode
        if max_batches is not None and batch_count >= max_batches:
            break
        
        if max_time and (time.time() - start) > 60*max_time: 
            break
        
        # Start timing this batch
        batch_start_time = time.time()
            
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
        
        total_loss += float(loss)
        total_count += 1
        
        # Store batch accuracy
        batch_acc = log_dict['acc'].mean().item()
        all_acc.append(batch_acc)

        if optimizer:
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
            
        # Calculate and print the time taken for this batch
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_count} processing time: {batch_time:.4f} seconds")
        
        batch_count += 1        
        t.set_description(f"{total_loss/total_count:.8f}")
        
        # Log metrics to wandb
        run.log({
            "loss": total_loss / total_count,
            "accuracy": batch_acc,
            "batch_time": batch_time
        })
        
    avg_acc = sum(all_acc) / len(all_acc) if all_acc else 0
    return total_loss / total_count, avg_acc

def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss, best_path = float('inf'), None

    # Determine max batches for debug mode
    train_batches = 10 if args.debug else None
    val_batches = 5 if args.debug else None

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = loop(train_loader, model, optimizer=optimizer, 
                                     max_time=args.train_time, max_batches=train_batches)
        
        path = args.model_path + f'RES_{model_id}_{epoch}.pt'
        torch.save(model.state_dict(), path)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.8f} acc: {train_acc:.2f}%')

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(val_loader, model, max_time=args.val_time, 
                                     max_batches=val_batches)
        print(f'\nEPOCH {epoch} VAL loss: {val_loss:.8f} acc: {val_acc:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = path
        print(f'BEST {best_path} VAL loss: {best_val_loss:.8f}')

        # Log metrics to wandb
        run.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

def test(model, test_loader):
    model.load_state_dict(torch.load(args.test))
    model.eval()
    
    # Determine max batches for debug mode
    test_batches = 5 if args.debug else None
    
    with torch.no_grad():
        test_loss, test_acc = loop(test_loader, model, max_batches=test_batches)
    
    print(f'\nTEST loss: {test_loss:.8f} acc: {test_acc:.2f}%')

    # Log metrics to wandb
    run.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })

def forward(model, batch, device):
    batch = batch.to(device)
    return model(batch)

def get_metrics():
    return {'accuracy': metrics.accuracy}

def main():
    # Adjust parameters for debug mode
    if args.debug:
        print("Running in DEBUG mode")
        max_samples = 100  # Limit dataset size
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

    # Initialize model with correct dimensions
    model = ProteinGrid(
        node_types=9,  # Number of atom types
        res_types=21,  # Number of residue types
        on_bb=2,      # On backbone indicator
        hidden_features=args.hidden_nf,
        out_features=20,  # Number of amino acid classes
    ).to(device)

    if args.test:
        test(model, test_loader)
    else:
        train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
    # Finish the run and upload any remaining data
    run.finish()