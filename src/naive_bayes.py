"""
Requirements:
- Binarize each pixel (1 if pixel > 0.5, else 0).
- Estimate conditional probabilities of each pixel being “on” given a digit class.
- Apply Bayes’ rule with the independence assumption.

"""
import numpy as np
import time

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    print("\n==============================")
    print("Running Naive Bayes Classifier (NumPy Only)")
    print("==============================")

    start_time = time.time()

    num_classes = 10
    num_features = X_train.shape[1]

    # Compute prior P(y)
    class_priors = np.zeros(num_classes)
    for c in range(num_classes):
        class_priors[c] = np.mean(y_train == c)

    # Conditional probabilities P(x_i=1 | y)
    cond_probs = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        X_c = X_train[y_train == c]
        cond_probs[c, :] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)  # Laplace smoothing

    # Compute log probabilities for numerical stability
    log_priors = np.log(class_priors)
    log_cond = np.log(cond_probs)
    log_cond_inv = np.log(1 - cond_probs)

    # Predict
    y_pred = []
    for x in X_test:
        log_probs = log_priors + np.sum(x * log_cond + (1 - x) * log_cond_inv, axis=1)
        y_pred.append(np.argmax(log_probs))
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    end_time = time.time()

    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("==============================\n")

    return accuracy
