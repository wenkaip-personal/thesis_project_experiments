import argparse

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

args = parser.parse_args()

import torch
from torch import nn, optim
import tqdm
import time
from functools import partial
from atom3d.util import metrics
from dataset import RESDataset
from model import ResEnTransformer
print = partial(print, flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())

def loop(dataset, model, optimizer=None, max_time=None):
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    t = tqdm.tqdm(dataset)
    metrics = get_metrics()
    total_loss, total_count = 0, 0
    targets, predicts = [], []

    for batch in t:
        if max_time and (time.time() - start) > 60*max_time: 
            break
            
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
        
        # Convert the label to a tensor with long dtype for classification
        batch_label = torch.tensor([batch.label], device=device, dtype=torch.long)
        # Add batch dimension to output
        out_batched = out.unsqueeze(0)
        loss_value = loss_fn(out_batched, batch_label)
        
        total_loss += float(loss_value)
        total_count += 1
        
        # Adjust prediction to match batched format
        pred_class = out_batched.argmax(dim=-1)
        targets.append(batch.label)
        predicts.extend(list(pred_class.cpu().numpy()))

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
                
        t.set_description(f"Loss: {total_loss/total_count:.8f}")
        
    accuracy = metrics['accuracy'](targets, predicts)
    return total_loss / total_count, accuracy

def train(model, train_dataset, val_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss, best_path = float('inf'), None

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = loop(train_dataset, model, optimizer=optimizer, max_time=args.train_time)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.8f} acc: {train_acc:.2f}%')

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loop(val_dataset, model, max_time=args.val_time)
        print(f'\nEPOCH {epoch} VAL loss: {val_loss:.8f} acc: {val_acc:.2f}%')

        path = f'res_en_transformer_baseline/models/RES_{model_id}_{epoch}.pt'
        torch.save(model.state_dict(), path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = path
        print(f'BEST {best_path} VAL loss: {best_val_loss:.8f}')

def test(model, test_dataset):
    model.load_state_dict(torch.load(args.test))
    model.eval()
    
    with torch.no_grad():
        test_loss, test_acc = loop(test_dataset, model)
    
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test loss: {test_loss:.8f}")

def forward(model, batch, device):
    batch = batch.to(device)
    
    # Pass the batch to the model - now the model will handle masking internally
    return model(batch.atoms, batch.x, batch.edge_index, batch)

def get_metrics():
    return {'accuracy': metrics.accuracy}

def main():
    # Load datasets
    data_path = args.data_path
    split_path = args.split_path
    dataset = partial(RESDataset, data_path) 
    train_dataset = dataset(split_path=split_path + 'train_indices.txt')
    val_dataset = dataset(split_path=split_path + 'val_indices.txt')
    test_dataset = dataset(split_path=split_path + 'test_indices.txt')

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