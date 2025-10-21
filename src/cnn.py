"""
Requirements:
- At least two convolutional layers with activation + pooling.
- Example: Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → FC → Softmax.
- Train with cross-entropy loss.
"""
# src/cnn.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================
# Convolutional Neural Network (CNN)
# ==========================
# Architecture:
# Conv(1→32, 3×3) → ReLU → MaxPool
# Conv(32→64, 3×3) → ReLU → MaxPool
# FC (3136 → 128) → ReLU → FC (128 → 10)
# Trained using CrossEntropyLoss (includes softmax internally)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1st convolutional block: input channel=1 (grayscale), output=32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # reduces image size by 2x

        # 2nd convolutional block: 32→64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two poolings: 28x28 → 14x14 → 7x7, so feature map size = 7x7
        # 64 filters × 7 × 7 = 3136 features total
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # fully connected layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0–9)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool2(self.relu2(self.conv2(x)))  # Conv2 + ReLU + Pool

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)  # reshape to (batch_size, 3136)

        # Fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)  # logits (raw scores)
        return x


# ==========================
# CNN Classifier Function
# ==========================
def cnn_classifier(X_train, X_test, y_train, y_test, epochs=5, batch_size=64, lr=0.01):
    """
    Train and evaluate a simple Convolutional Neural Network (CNN) on MNIST.

    Parameters
    ----------
    X_train : np.ndarray
        Training images, shape (N, 1, 28, 28), normalized to [0, 1].
    X_test : np.ndarray
        Test images, same format as X_train.
    y_train : np.ndarray
        Integer labels for training images (0–9).
    y_test : np.ndarray
        Integer labels for test images (0–9).
    epochs : int, optional
        Number of training epochs. Default is 5.
    batch_size : int, optional
        Mini-batch size for DataLoader. Default is 64.
    lr : float, optional
        Learning rate for SGD optimizer. Default is 0.01.

    Returns
    -------
    None
        Prints training progress and final test accuracy.
    """

    print("\n==============================")
    print("Running CNN Classifier (PyTorch)")
    print("==============================")

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create PyTorch datasets and data loaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device setup (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # automatically applies Softmax
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start_time = time.time()

    # --------------------------
    # Training loop
    # --------------------------
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          # reset gradients
            outputs = model(images)        # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()                # backpropagation
            optimizer.step()               # update weights

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    # --------------------------
    # Evaluation on test data
    # --------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # choose class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nCNN Accuracy: {accuracy:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("==============================\n")
