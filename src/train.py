import torch
import torch.nn as nn
import wandb
import os
from model import get_densenet, get_vit

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
 
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)                          # Move data to whatever device user is using
        labels = labels.to(device)

        optimizer.zero_grad()                               # Zero out gradients so they don't accumulate by default

        outputs = model(images)                             # Foward pass: pass images through the model to get predictions

        loss = criterion(outputs, labels)                   # Calculate loss: compare predictions to true labels

        loss.backward()                                     # Backward pass: calculate gradients; how much did each weight cont. to the error?

        optimizer.step()                                    # Update weights: use the optimizer to adjust weights based on gradients

        running_loss += loss.item()                         # Track loss and accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)             # Return metrics
    accuracy = correct / total
    return avg_loss, accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)                          # Move data to whatever device user is using
            labels = labels.to(device)
                
            outputs = model(images)                             # Foward pass: pass images through the model to get predictions
            
            loss = criterion(outputs, labels)                   # Calculate loss: compare predictions to true labels
                        
            running_loss += loss.item()                         # Track loss and accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)             # Return metrics
    accuracy = correct / total
    return avg_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, patience):
    wandb.init(project="alzheimer-mri-progression", config={
        "epochs": num_epochs,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "architecture": "DenseNet121"
    })

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):                       # loop runs num_epoch times

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss: .4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"New best model saved with val accuracy: {val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1

        if epochs_no_improve >= patience:
            print (f"Early stopping triggered after {epoch+1} epochs")
            break

        scheduler.step()

