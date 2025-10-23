"""
Provides utilities for analyzing and visualizing ML model performance.
Includes:
1. Confusion matrix visualization
4. Visualizations for Linear Classifier and Naïve Bayes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn


'''
1. Confusion Matrix Visualization
'''

def show_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Displays confusion matrix for classification results.
    Helps identify which classes are most often confused.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()


'''
4. Visualization Utilities
'''
def visualize_linear_weights(W):
    """
    Visualizes the learned weights of a linear classifier (e.g., 784 x 10 for MNIST).
    Each column of W corresponds to one digit class.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        # take column i instead of row i
        ax.imshow(W[:, i].reshape(28, 28), cmap="coolwarm")
        ax.set_title(f"Digit {i}")
        ax.axis("off")
    plt.suptitle("Weight Visualization for Linear Classifier")
    plt.show()



def visualize_naive_bayes_probs(class_probs):
    """
    Displays the probability map for each class (e.g., Naïve Bayes pixel likelihoods).
    Each image shows which pixels are most likely active for that class.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(class_probs[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Class {i}")
        ax.axis("off")
    plt.suptitle("Naïve Bayes Conditional Probabilities")
    plt.show()
