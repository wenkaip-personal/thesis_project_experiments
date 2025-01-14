import torch
from data import get_res_dataloaders
from model import RESEGNN
from train_eval import train_model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    N_LAYERS = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    # Paths to the split datasets
    TRAIN_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/train"
    VAL_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/val"
    TEST_PATH = "/content/drive/MyDrive/thesis_project/atom3d_res_dataset/split-by-cath-topology/data/test"
    
    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_res_dataloaders(
        TRAIN_PATH, VAL_PATH, TEST_PATH,
        batch_size=BATCH_SIZE
    )
    print("Datasets loaded successfully")
    
    # Get number of node features from first batch
    for batch in train_loader:
        in_node_nf = batch.x.size(-1)  # Get number of input features
        break
        
    # Initialize model with correct input dimension
    print("Initializing model...")
    model = RESEGNN(
        in_node_nf=in_node_nf,  # Use detected input dimension
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
    main()