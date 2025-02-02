import torch
from data import get_res_dataloaders
from model import RESEGNN
from train_eval import train_model

def main(debug=True):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters - reduced for debug mode
    BATCH_SIZE = 8 if debug else 32
    HIDDEN_DIM = 64 if debug else 128 
    N_LAYERS = 2 if debug else 4
    LEARNING_RATE = 1e-4
    EPOCHS = 2 if debug else 100
    
    # Paths to the split datasets
    TRAIN_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/train"
    VAL_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/val" 
    TEST_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/test"
    
    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_res_dataloaders(
        TRAIN_PATH, VAL_PATH, TEST_PATH,
        batch_size=BATCH_SIZE,
        debug=debug
    )
    print("Datasets loaded successfully")
    
    # Initialize model
    print("Initializing model...")
    model = RESEGNN(
        hidden_nf=HIDDEN_DIM,
        n_layers=N_LAYERS,
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
    # Run with debug=True first
    main(debug=True)