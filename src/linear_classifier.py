"""
Linear Classifier — NumPy Implementation
Requirements:
- Implement a linear classifier: y = XW
- Train using L2 loss and gradient descent
- No PyTorch (NumPy only)
- Return predictions + weights for visualization
"""

import numpy as np
import time


def linear_classifier(X_train, X_test, y_train, y_test, epochs=100, lr=0.01):
    '''
    Trains and evaluates a linear classifier on MNIST digits.
    Uses gradient descent to minimize L2 loss.
    Returns:
        y_pred_labels: predicted test labels
        W: learned weight matrix (for visualization)
    '''
    print("\n==============================")
    print("Running Linear Classifier (NumPy Only)")
    print("==============================")

    start_time = time.time()

    # --- Setup dimensions ---
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    # --- One-hot encode labels ---
    '''
    Convert labels (0–9) to one-hot matrix for loss computation.
    Example: label 3 → [0,0,0,1,0,0,0,0,0,0]
    '''
    Y_train = np.zeros((n_samples, n_classes))
    Y_train[np.arange(n_samples), y_train] = 1

    # --- Initialize weights ---
    '''
    Small random values; shape: (784, 10)
    '''
    W = np.random.randn(n_features, n_classes) * 0.01

    # --- Training loop ---
    for epoch in range(epochs):
        # Forward pass
        Y_pred = X_train @ W  # predictions (N x 10)

        # Compute L2 loss
        loss = np.mean((Y_pred - Y_train) ** 2)

        # Backpropagation (gradient)
        grad = (2 / n_samples) * X_train.T @ (Y_pred - Y_train)

        # Gradient descent update
        W -= lr * grad

        # Print progress occasionally
        if (epoch + 1) % (epochs // 5) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    # --- Evaluation phase ---
    '''
    Predict test labels by computing X_test @ W and taking argmax.
    '''
    Y_test_pred = X_test @ W
    y_pred_labels = np.argmax(Y_test_pred, axis=1)
    accuracy = np.mean(y_pred_labels == y_test)

    end_time = time.time()
    print(f"\nLinear Classifier Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("==============================\n")

    return y_pred_labels, W
