"""
Requirements:
- At least one hidden layer with nonlinearity (e.g., ReLU).
- Train with SGD
Example: 784 → 256 → 128 → 10.
ChatGPT was used to figure out how to include a model with Pytorch initization and steps.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define MLP model according to: 784 → 256 → 128 → 10
class MLP(nn.Module):
    """
    Basic NN:
    - Input layer 784 nodes, 1 per pixel
    - Two hidden layers with ReLU activations
    - Output layer 10 nodes, 1 per digit
    """
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(MLP, self).__init__()

        # 1st layer 784 -> 256
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        # 2nd layer 256 -> 128
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        # final layer: 128 -> 10 (digits 0-9)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        '''
        How data moves through layers:
        input -> hidden layer 1 -> ReLU -> Hidden layer 2 -> ReLU -> Output
        Forward pass:
        the input goes through each layer in sequence,
        applying ReLU activation in between

        '''
        x = self.fc1(x) 
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def mlp_classifier(X_train, X_test, y_train, y_test, epochs=20, lr=0.01, batch_size=64):
    '''
    Trains and evaluates the MLP model using SGD and CrossEntropyLoss.

    Parameters:
      X_train, y_train — training data and labels
      X_test, y_test   — test data and labels
      epochs           — number of training loops
      lr               — learning rate
      batch_size       — number of images per training batch
    '''

    # do we have GPU?
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
    criterion = nn.CrossEntropyLoss() # loss for classification 
    optimizer = optim.SGD(model.parameters(), lr=lr) # basic optimizer

    start_time = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train() # model in training mode
        permutation = torch.randperm(X_train.size(0)) # shuffle data for each epoch
        total_loss = 0.0

        # loop through training data in batches
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            # 1: reset gradients
            optimizer.zero_grad()
            # 2. forward pass (predict)
            outputs = model(batch_X) # calling model.forward(batch_X)
            # 3. compute loss
            loss = criterion(outputs, batch_y)
            # 4. compute gradients
            loss.backward()
            # 5. update weights
            optimizer.step()
            # track total loss 
            total_loss += loss.item() * batch_X.size(0)

        # print progress for very epochs
        avg_loss = total_loss / X_train.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval() # switch model to evalutation model 
    with torch.no_grad():
        outputs = model(X_test) # run model on test images
        _, preds = torch.max(outputs, 1) # pick label w/ highest score
        accuracy = (preds == y_test).float().mean().item() 

    elapsed = time.time() - start_time
    print(f"\nMLP Accuracy: {accuracy:.4f}")
    print(f"Time taken: {elapsed:.2f}s")
    print("==============================")

    # return: predictions for confusion matrix
    return preds.cpu().numpy()
