"""
Linear Classifier — NumPy Implementation
ChatGPT was used to figure out how to steps on how to create function linear_classifier().
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
    Trains and evaluates a linear classifier on MNIST digits. It tries to find the best weight matrix Weight so that 
    Uses gradient descent to minimize L2 loss.
    Returns:
        y_pred_labels: predicted test labels
        W: learned weight matrix (for visualization)
    '''
    print("\n==============================")
    print("Running Linear Classifier (NumPy Only)")
    print("==============================")

    start_time = time.time()

    # setup dimensions
    n_samples, n_features = X_train.shape # how many images and pixels 
    n_classes = len(np.unique(y_train)) # number of digits (0-9)

    '''
    1. Convert labels (0-9) into "one-hot"
        e.g. if the digit = 3 then [0,0,0,1,0,0,0,0,0,0]
    Convert labels (0–9) to one-hot matrix for loss computation.
    Example: label 3 → [0,0,0,1,0,0,0,0,0,0]
    '''
    Y_train = np.zeros((n_samples, n_classes))
    Y_train[np.arange(n_samples), y_train] = 1

    '''
    2. We initialize weights with small random numbers
    Each pixel will have a wight for each digit
    Small random values; shape: (784 pixels, 10 digits)
    '''
    W = np.random.randn(n_features, n_classes) * 0.01

    '''
    The linear classifier will update its weight many times epochs to make predictions closer to real labels
    '''
    for epoch in range(epochs):
        # 3. Forward pass (e.g. predict outputs)
        Y_pred = X_train @ W  # predictions (N x 10)

        # 4. Compute L2 loss (e.g. compute error)
        loss = np.mean((Y_pred - Y_train) ** 2)

        # 5. Find out how to adjust weights (gradient)
        grad = (2 / n_samples) * X_train.T @ (Y_pred - Y_train)

        # 6. Update weights -> Gradient descent update
        W -= lr * grad

        # Print progress in terminal
        if (epoch + 1) % (epochs // 5) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    '''
    7. Predict test labels by computing X_test @ W and taking argmax.
    (e.g. use the learned weights to make predictions on test images)
    Choose the class with the highest score for each image
    '''
    Y_test_pred = X_test @ W
    y_pred_labels = np.argmax(Y_test_pred, axis=1)

    # 8. compare predictions with real labels to get accuracy
    accuracy = np.mean(y_pred_labels == y_test)

    end_time = time.time()
    print(f"\nLinear Classifier Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("==============================\n")

    '''
    9. return:
    y_pred_labels: what model guess for each image
    W: learned weights for visualization
    '''
    return y_pred_labels, W
