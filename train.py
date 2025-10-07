import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules.data_handler import load_hyperspectral_data
from modules.model_handler import prepare_training_data, CropClassifier, PATCH_SIZE

# --- Configuration ---
DATA_PATH = 'data'
MODEL_SAVE_PATH = os.path.join('models', 'crop_classifier.pth') # PyTorch models typically .pth

def main():
    """Main function to execute the training pipeline."""
    print("--- Starting PyTorch Model Training Process ---")

    # 1. Load Data
    print("Step 1/5: Loading hyperspectral data...")
    try:
        hypercube, ground_truth = load_hyperspectral_data(DATA_PATH)
        print(f"Data loaded successfully. Hypercube shape: {hypercube.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the Indian Pines dataset files are in the 'data' directory.")
        return

    # 2. Prepare Data for Training
    print("Step 2/5: Preparing data for training...")
    X, y = prepare_training_data(hypercube, ground_truth)
    print(f"Data prepared. Number of patches: {len(X)}")

    # 3. Split Data
    print("Step 3/5: Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. Create Model, Loss Function, and Optimizer
    print("Step 4/5: Creating model, loss function, and optimizer...")
    num_classes = len(torch.unique(y))
    model = CropClassifier(num_classes=num_classes)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Train Model
    print("Step 5/5: Starting model training...")
    print("This may take a significant amount of time depending on your hardware.")
    
    num_epochs = 30 # Reduced for faster demonstration
    best_loss = float('inf')
    patience = 5 # For early stopping
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item()
        
        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%')

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break

    print("--- PyTorch Model Training Complete ---")
    print(f"The best model has been saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()