import argparse
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
import os
import torch_geometric
from functools import partial
from atom3d.util import metrics
from dataset_grid import GridRESDataset
from model_grid import EquivariantGridModel

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='number of threads for loading data')
parser.add_argument('--batch', metavar='SIZE', type=int, default=4,
                    help='batch size')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time per training on trainset')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=5e-4, type=float,
                    help='learning rate')
parser.add_argument('--hidden_nf', type=int, default=128,
                    help='number of hidden features')
parser.add_argument('--grid_size', type=int, default=6,
                    help='size of grid (grid_size x grid_size x grid_size)')
parser.add_argument('--spacing', type=float, default=1.2,
                    help='spacing between grid points')
parser.add_argument('--k', type=int, default=5,
                    help='number of nearest neighbors for grid connections')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay for regularization')
parser.add_argument('--data_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/')
parser.add_argument('--split_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/')
parser.add_argument('--model_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/thesis_project_experiments/res_equi_grid/models/')
# Add debug mode flag
parser.add_argument('--debug', action='store_true',
                    help='enable debug mode with reduced dataset size and fewer epochs')

args = parser.parse_args()

print = partial(print, flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())

# Make sure model directory exists
os.makedirs(args.model_path, exist_ok=True)

# Start a new wandb run to track this script
# run = wandb.init(
#     project="res",
#     config={
#         "learning_rate": args.lr,
#         "architecture": "ResEquiGrid",
#         "dataset": "RES",
#         "epochs": args.epochs,
#         "grid_size": args.grid_size,
#         "spacing": args.spacing,
#         "weight_decay": args.weight_decay,
#     },
# )

def loop(dataset, model, optimizer=None, max_time=None, max_batches=None):
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    t = tqdm.tqdm(dataset)
    metrics_dict = get_metrics()
    total_loss, total_count = 0, 0
    targets, predicts = [], []

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
            # Move batch to device and run forward pass
            batch = batch.to(device)
            out = model(batch.atoms, batch.x, batch.edge_index, batch)
            
            # Get the label from batch
            label = batch.label
            
            # Compute loss
            loss_value = loss_fn(out, label)
            
            total_loss += float(loss_value)
            total_count += 1
            
            # Get predictions
            pred_class = out.argmax(dim=-1)
            targets.extend(label.cpu().tolist())
            predicts.extend(pred_class.cpu().tolist())

            if optimizer:
                try:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    loss_value.backward()
                    optimizer.step()
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise(e)
                    torch.cuda.empty_cache()
                    print('Skipped batch due to OOM', flush=True)
                    continue
                
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print(f"Error: {str(e)}")
                raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
            
        # Calculate and print the time taken for this batch
        batch_time = time.time() - batch_start_time
        
        batch_count += 1        
        t.set_description(f"Loss: {total_loss/total_count:.8f}")
        
        # Log metrics to wandb
        # run.log({
        #     "loss": total_loss / total_count,
        #     "accuracy": metrics_dict['accuracy'](targets, predicts),
        #     "batch_time": batch_time
        # })
        
    accuracy = metrics_dict['accuracy'](targets, predicts)
    return total_loss / total_count, accuracy

def train(model, train_dataset, val_dataset):
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with more appropriate parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    best_val_loss, best_path = float('inf'), None

    # Determine max batches for debug mode
    train_batches = 10 if args.debug else None
    val_batches = 5 if args.debug else None

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_acc = loop(train_dataset, model, optimizer=optimizer, 
                                     max_time=args.train_time, max_batches=train_batches)
        
        # Save model
        path = args.model_path + f'RES_GRID_{model_id}_{epoch}.pt'
        torch.save(model.state_dict(), path)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.8f} acc: {train_acc:.2f}%')

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(val_dataset, model, max_time=args.val_time, 
                                     max_batches=val_batches)
        print(f'\nEPOCH {epoch} VAL loss: {val_loss:.8f} acc: {val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = path
            # Save best model separately
            torch.save(model.state_dict(), args.model_path + 'best_model.pt')
        print(f'BEST {best_path} VAL loss: {best_val_loss:.8f}')

        # Log metrics to wandb
        # run.log({
        #     "train_loss": train_loss,
        #     "train_acc": train_acc,
        #     "val_loss": val_loss,
        #     "val_acc": val_acc,
        #     "learning_rate": optimizer.param_groups[0]['lr']
        # })

def test(model, test_dataset):
    # Load best model if testing
    model.load_state_dict(torch.load(args.test))
    model.eval()
    
    # Determine max batches for debug mode
    test_batches = 5 if args.debug else None
    
    with torch.no_grad():
        test_loss, test_acc = loop(test_dataset, model, max_batches=test_batches)
    
    print(f'\nTEST loss: {test_loss:.8f} acc: {test_acc:.2f}%')

    # Log metrics to wandb
    # run.log({
    #     "test_loss": test_loss,
    #     "test_acc": test_acc
    # })

def get_metrics():
    return {'accuracy': metrics.accuracy}

def main():
    # Adjust parameters for debug mode
    if args.debug:
        print("Running in DEBUG mode")
        max_samples = 100  # Limit dataset size
    else:
        max_samples = None
    
    # Load datasets
    data_path = args.data_path
    split_path = args.split_path
    dataset = partial(GridRESDataset, data_path,
                     grid_size=args.grid_size, spacing=args.spacing, k=args.k)
    
    train_dataset = dataset(split_path=split_path + 'train_indices.txt', max_samples=max_samples)
    val_dataset = dataset(split_path=split_path + 'val_indices.txt', max_samples=max_samples)
    test_dataset = dataset(split_path=split_path + 'test_indices.txt', max_samples=max_samples)

    datasets = train_dataset, val_dataset, test_dataset
    
    dataloader = partial(torch_geometric.loader.DataLoader, 
                        num_workers=args.num_workers, 
                        batch_size=args.batch,
                        follow_batch=['grid_coords'])  # Track batch for grid coordinates

    train_dataset, val_dataset, test_dataset = map(dataloader, datasets)

    # Initialize model with correct dimensions
    model = EquivariantGridModel(
        in_node_nf=9,  # One-hot encoded atoms (9 possible elements)
        hidden_nf=args.hidden_nf,
        out_node_nf=20,  # Number of amino acid classes
        grid_size=args.grid_size,
        device=device
    ).to(device)

    if args.test:
        test(model, test_dataset)
    else:
        train(model, train_dataset, val_dataset)

if __name__ == "__main__":
    main()
    # Finish the run and upload any remaining data
    # run.finish()