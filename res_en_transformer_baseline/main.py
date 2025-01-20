import torch
from data import get_res_dataloaders
from model import RESEnTransformer
from train_eval import train_model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    N_LAYERS = 4
    HEADS = 8
    DIM_HEAD = 32
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    # Paths to the split datasets 
    TRAIN_PATH = "data/split-by-cath-topology/train"
    VAL_PATH = "data/split-by-cath-topology/val"  
    TEST_PATH = "data/split-by-cath-topology/test"
    
    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_res_dataloaders(
        TRAIN_PATH, VAL_PATH, TEST_PATH,
        batch_size=BATCH_SIZE
    )
    print("Datasets loaded successfully")
    
    # Initialize model 
    print("Initializing model...")
    model = RESEnTransformer(
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        heads=HEADS,
        dim_head=DIM_HEAD,
        dropout=DROPOUT,
        device=device
    )
    print("Model initialized successfully")
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )

if __name__ == "__main__":
    main()