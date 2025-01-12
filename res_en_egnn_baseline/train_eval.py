import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move batch to device
        node_feats = batch['node_feats'].to(device)
        pos = batch['pos'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attrs = batch['edge_attrs'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(node_feats, pos, edge_index, edge_attrs)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1),
                         'acc': total_correct / total_samples})
    
    return total_loss / len(train_loader), total_correct / total_samples

def evaluate(model, data_loader, criterion, device, split="val"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'{split} Evaluation')
        for batch in pbar:
            # Move batch to device
            node_feats = batch['node_feats'].to(device)
            pos = batch['pos'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attrs = batch['edge_attrs'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(node_feats, pos, edge_index, edge_attrs)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1),
                            'acc': total_correct / total_samples})
    
    return total_loss / len(data_loader), total_correct / total_samples

def train_model(model, train_loader, val_loader, test_loader, 
                epochs=100, lr=1e-4, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    best_model = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                          criterion, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "val")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "test")
    print(f"\nFinal Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")