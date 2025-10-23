"""
K-Nearest Neighbors (KNN) — NumPy Only
======================================
Requirements:
- Load MNIST directly from raw image files.
- Create your own training/testing partitions.
- Represent each image as a flattened 784-dimensional vector.
- Compute Euclidean distance.
- Classify by majority vote among k neighbors.
- Test with at least k = 1, 3, 5.
- No scikit-learn KNeighborsClassifier.
"""

import numpy as np
import time
from scipy.stats import mode


def compute_euclidean_distance(X_train, X_test):
    '''
    Compute Euclidean distance between all test and train samples.
    Uses vectorized math for efficiency:
        (x - y)^2 = x^2 + y^2 - 2xy
    Returns:
        distances: matrix (num_test, num_train)
    '''
    X_train_sq = np.sum(X_train ** 2, axis=1)
    X_test_sq = np.sum(X_test ** 2, axis=1)
    distances = np.sqrt(
        X_test_sq[:, np.newaxis] + X_train_sq[np.newaxis, :] - 2 * np.dot(X_test, X_train.T)
    )
    return distances


def knn_predict(X_train, y_train, X_test, k=3):
    '''
    Predict labels for X_test using KNN.
    For each test sample:
      - Compute distances to all training samples
      - Select k nearest neighbors
      - Use majority vote of their labels
    Returns:
        preds: predicted labels for X_test
    '''
    distances = compute_euclidean_distance(X_train, X_test)
    neighbors_idx = np.argsort(distances, axis=1)[:, :k]
    neighbors_labels = y_train[neighbors_idx]
    preds, _ = mode(neighbors_labels, axis=1)
    return preds.ravel()


def knn_classifier(X_train, X_test, y_train, y_test, k_values=[1, 3, 5]):
    '''
    Evaluate KNN classifier for multiple k values.
    For each k:
      - Predict test labels
      - Compute accuracy and runtime
    Returns:
        preds: predictions for the last tested k
    '''
    print("\n==============================")
    print("Running KNN Classifier (NumPy Only)")
    print("==============================")

    last_preds = None
    for k in k_values:
        start_time = time.time()
        preds = knn_predict(X_train, y_train, X_test, k)
        acc = np.mean(preds == y_test)
        end_time = time.time()

        runtime = end_time - start_time
        print(f"k = {k} | Accuracy = {acc:.4f} | Time = {runtime:.2f}s")

        last_preds = preds  # save last predictions

    print("==============================\n")
    return last_preds  # return predictions for analysis
