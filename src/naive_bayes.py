"""
Naive Bayes Classifier — NumPy Only
This code was created with the help of ChatGPT 5.0. Steps and errors were assisted.
Requirements:
- Binarize each pixel (1 if pixel > 0.5, else 0).
- Estimate conditional probabilities of each pixel being “on” given a digit class.
- Apply Bayes’ rule with the independence assumption.
- Return predictions and conditional probabilities for analysis.
"""

import numpy as np
import time


"""
X_train: input data (images)
y_train: labels (correct answers)
X_test: input data (features, new images the classifiers has never seen)
y_test: labels (correct answers for test images, used for accuracy)
"""
def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    print("\n==============================")
    print("Running Naive Bayes Classifier (NumPy Only)")
    print("==============================")

    start_time = time.time()

    num_classes = 10 # 10 digits
    num_features = X_train.shape[1] # each image has 784 pixels (28x28)

    # 1. Compute prior P(y)
    # We can say how often does each digit appear in the training set
    class_priors = np.zeros(num_classes)
    for c in range(num_classes):
        class_priors[c] = np.mean(y_train == c) # find the num of samples that are in class c



    # 2. Conditional probabilities P(x_i=1 | y)
    # For each digit (class), how likely is each pixel "on"
    # CHATGPT helps with smoothing to AVOID divide by 0
    cond_probs = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        X_c = X_train[y_train == c]
        cond_probs[c, :] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)  # Laplace smoothing

    # 3. Compute log probabilities for stability
    # Probabilies can be small, so logs is used to make them manageble. 
    log_priors = np.log(class_priors)
    log_cond = np.log(cond_probs)
    log_cond_inv = np.log(1 - cond_probs)

    # 4. Predict test samples
    # For each image, we calculate the score for each digit (class)
    # we then pick the digit (0-9) with the highest porbability
    y_pred = []
    for x in X_test:
        # independence assumption - ChatGPT helped
        log_probs = log_priors + np.sum(x * log_cond + (1 - x) * log_cond_inv, axis=1)
        y_pred.append(np.argmax(log_probs))
    y_pred = np.array(y_pred)

    # 5. Compute accuracy
    # view how many predictions were correct
    accuracy = np.mean(y_pred == y_test)
    end_time = time.time()

    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print("==============================\n")

    # 6. Return predictions and conditional probabilities
    # y_pred: the classifier's guesses for each image
    # cond_probs: how likely each pixel "on" for all images
    return y_pred, cond_probs
