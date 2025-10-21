"""
Requirements:
- Implement a linear classifier: y = Wx.
- Train using L2 loss and gradient descent.
Options:
- NumPy: write your own forward, backward, and update loop.
- PyTorch: use nn.Linear with optimizer and autograd.

"""
import numpy as np
import time

def linear_classifier(X_train, X_test, y_train, y_test, epochs=100, lr=0.01):
    """
    Linear classifier: y = X W
    Train with L2 loss and gradient descent.
    """
    print("\n==============================")
    print("Running Linear Classifier (NumPy Only)")
    print("==============================")
    start_time = time.time()
    
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    
    # One-hot encode labels
    Y_train = np.zeros((n_samples, n_classes))
    Y_train[np.arange(n_samples), y_train] = 1
    
    # Initialize weights
    W = np.random.randn(n_features, n_classes) * 0.01
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass: compute predictions
        Y_pred = X_train @ W  # shape (n_samples, n_classes)
        
        # Compute L2 loss
        loss = np.mean((Y_pred - Y_train)**2)
        
        # Backward pass: gradient
        grad = (2/n_samples) * X_train.T @ (Y_pred - Y_train)
        
        # Gradient descent update
        W -= lr * grad
        
        if (epoch+1) % (epochs//5) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
    
    # Evaluation
    Y_test_pred = X_test @ W
    y_pred_labels = np.argmax(Y_test_pred, axis=1)
    accuracy = np.mean(y_pred_labels == y_test)
    
    end_time = time.time()
    print(f"\nLinear Classifier Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("==============================\n")
    return accuracy
