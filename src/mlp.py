"""
Requirements:
- At least one hidden layer with nonlinearity (e.g., ReLU).
- Train with SGD
Example: 784 → 256 → 128 → 10.

"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define MLP model according to: 784 → 256 → 128 → 10
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def mlp_classifier(X_train, X_test, y_train, y_test, epochs=20, lr=0.01, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==============================")
    print("Running MLP Classifier (PyTorch)")
    print("==============================")
    print(f"Using device: {device}")

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test, dtype=torch.long).to(device)

    # Initialize model, loss function, and optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    start_time = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / X_train.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test).float().mean().item()

    elapsed = time.time() - start_time
    print(f"\nMLP Accuracy: {accuracy:.4f}")
    print(f"Time taken: {elapsed:.2f}s")
    print("==============================")
    return preds.cpu().numpy()
