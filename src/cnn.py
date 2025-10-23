"""
Convolutional Neural Network (CNN) — PyTorch Implementation
============================================================
Requirements:
- Two convolutional layers with activation + pooling.
- Example: Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC → Softmax.
- Train with CrossEntropyLoss.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


'''
Define CNN Architecture
'''
class SimpleCNN(nn.Module):
    '''
    A simple convolutional neural network for MNIST classification.

    Architecture:
        Conv(1→32, 3×3) → ReLU → MaxPool
        Conv(32→64, 3×3) → ReLU → MaxPool
        FC (3136→128) → ReLU → FC (128→10)
    '''
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        '''
        Forward pass through CNN.
        Each convolutional block extracts spatial features.
        '''
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 3136)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


'''
CNN Training and Evaluation
'''
def cnn_classifier(X_train, X_test, y_train, y_test, epochs=5, batch_size=64, lr=0.01):
    '''
    Trains a CNN on MNIST and returns predictions for confusion matrix.
    '''
    print("\n==============================")
    print("Running CNN Classifier (PyTorch)")
    print("==============================")

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start_time = time.time()

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")

    # --------------------------
    # Evaluation phase
    # --------------------------
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    print(f"\nCNN Accuracy: {accuracy:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("==============================\n")

    # Return predictions for confusion matrix
    return all_preds
