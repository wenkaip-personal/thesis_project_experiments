import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='number of threads for loading data')
parser.add_argument('--batch', metavar='SIZE', type=int, default=8,
                    help='batch size')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time per training on trainset')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--hidden_nf', type=int, default=128,
                    help='number of hidden features')
parser.add_argument('--n_layers', type=int, default=4,
                    help='number of graph conv layers')
parser.add_argument('--n_heads', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--data_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/raw/RES/data/')
parser.add_argument('--split_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/atom3d_res_dataset/indices/')
parser.add_argument('--model_path', type=str, 
                    default='/content/drive/MyDrive/thesis_project/thesis_project_experiments/res_en_transformer_baseline/models/')

# Add debug mode flag
parser.add_argument('--debug', action='store_true',
                    help='enable debug mode with reduced dataset size and fewer epochs')

args = parser.parse_args()

import torch
from torch import nn, optim
import tqdm
import time
import torch_geometric
from functools import partial
from atom3d.util import metrics
from dataset import RESDataset
from model import ResEnTransformer
print = partial(print, flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb project where this run will be logged.
    project="res-en-transformer",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": args.lr,
        "architecture": "ResEnTransformer",
        "dataset": "RES",
        "epochs": args.epochs,
    },
)

def loop(dataset, model, optimizer=None, max_time=None, max_batches=None):
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    t = tqdm.tqdm(dataset)
    print(len(t))
    metrics = get_metrics()
    total_loss, total_count = 0, 0
    targets, predicts = [], []

    batch_count = 0
    for batch in t:
        print(len(batch))

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
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM')
            continue
        
        # Use batch.label directly - it's already a batched tensor
        batch_label = batch.label.to(device)
        loss_value = loss_fn(out, batch_label)
        
        total_loss += float(loss_value)
        total_count += 1
        
        # Get predictions
        pred_class = out.argmax(dim=-1)
        targets.extend(batch_label.cpu().tolist())
        predicts.extend(pred_class.cpu().tolist())

        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM')
                continue
        
        # Calculate and print the time taken for this batch
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_count} processing time: {batch_time:.4f} seconds")
        
        batch_count += 1
        t.set_description(f"{total_loss/total_count:.8f}")

        # Log metrics to wandb.
        run.log({
            "loss": total_loss / total_count,
            "accuracy": metrics['accuracy'](targets, predicts)
        })
        
    accuracy = metrics['accuracy'](targets, predicts)
    return total_loss / total_count, accuracy

def train(model, train_dataset, val_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss, best_path = float('inf'), None

    # Determine max batches for debug mode
    train_batches = 10 if args.debug else None
    val_batches = 5 if args.debug else None

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = loop(train_dataset, model, optimizer=optimizer, 
                                     max_time=args.train_time, max_batches=train_batches)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.8f} acc: {train_acc:.2f}%')

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(val_dataset, model, max_time=args.val_time, 
                                     max_batches=val_batches)
        print(f'\nEPOCH {epoch} VAL loss: {val_loss:.8f} acc: {val_acc:.2f}%')

        path = args.model_path + f'RES_{model_id}_{epoch}.pt'
        torch.save(model.state_dict(), path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = path
        print(f'BEST {best_path} VAL loss: {best_val_loss:.8f}')

        # Log metrics to wandb.
        run.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

def test(model, test_dataset):
    model.load_state_dict(torch.load(args.test))
    model.eval()

    # Determine max batches for debug mode
    test_batches = 5 if args.debug else None
    
    with torch.no_grad():
        test_loss, test_acc = loop(test_dataset, model, max_batches=test_batches)
    
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test loss: {test_loss:.8f}")

    # Log metrics to wandb.
    run.log({
        "test_loss": test_loss,
        "test_acc": test_acc
    })

def forward(model, batch, device):
    batch = batch.to(device)

    # Pass the batch to the model - now the model will handle masking internally
    return model(batch.atoms, batch.x, batch.edge_index, batch)

def get_metrics():
    return {'accuracy': metrics.accuracy}

def main():
    # Adjust parameters for debug mode
    if args.debug:
        print("Running in DEBUG mode")
        args.epochs = min(args.epochs, 2)  # Reduce epochs for faster iteration
        args.hidden_nf = 32  # Reduce hidden dimension size
        args.n_layers = 2    # Reduce number of layers
    
    # Load datasets
    data_path = args.data_path
    split_path = args.split_path
    dataset = partial(RESDataset, data_path) 
    train_dataset = dataset(split_path=split_path + 'train_indices.txt')
    val_dataset = dataset(split_path=split_path + 'val_indices.txt')
    test_dataset = dataset(split_path=split_path + 'test_indices.txt')
    
    datasets = train_dataset, val_dataset, test_dataset
    print(len(train_dataset))
    dataloader = partial(torch_geometric.loader.DataLoader, num_workers=args.num_workers, batch_size=args.batch)

    train_dataset, val_dataset, test_dataset = map(dataloader, datasets)

    print(train_dataset[0])

    # Initialize model with correct dimensions
    model = ResEnTransformer(
        input_nf=9,  # One-hot encoded atoms (9 possible elements)
        output_nf=20,  # Number of amino acid classes
        hidden_nf=args.hidden_nf,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        device=device
    ).to(device)

    if args.test:
        test(model, test_dataset)
    else:
        train(model, train_dataset, val_dataset)

if __name__ == "__main__":
    main()
    # Finish the run and upload any remaining data.
    run.finish()