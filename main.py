"""

"""
from src.utils import prepare_all_versions
from src.knn import knn_classifier
from src.naive_bayes import naive_bayes_classifier
from src.linear_classifier import linear_classifier
from src.mlp import mlp_classifier
from src.cnn import cnn_classifier
from analysis import show_confusion_matrix, visualize_naive_bayes_probs, visualize_linear_weights


def main():
    # Load all versions of MNIST data
    data = prepare_all_versions()
    
    # Access train/test for each classifier
    knn_data = data["KNN"]
    nb_data = data["NaiveBayes"]
    linear_data = data["Linear"]
    mlp_data = data["MLP"]
    cnn_data = data["CNN"]
    

    print("\n==============================")
    print("Train/Test Split Info (All Versions)")
    print("==============================")

    for name, (X_train, X_test, y_train, y_test) in data.items():
        print(f"\n{name} Dataset:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape:  {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape:  {y_test.shape}")
        print(f"  Total samples: {len(X_train) + len(X_test)}")
        print(f"  Train/Test ratio: {len(X_train)/(len(X_train)+len(X_test)):.2f} / {len(X_test)/(len(X_train)+len(X_test)):.2f}")


    # Call each classifier

    # KNN
    y_pred_knn = knn_classifier(*knn_data, k_values=[1, 3, 5])
    # Show confusion matrix (Failure mode analysis)
    show_confusion_matrix(knn_data[3], y_pred_knn, "KNN Confusion Matrix")



    # Naive Bayes
    y_pred_nb, nb_cond_probs = naive_bayes_classifier(*nb_data)
    # Show confusion matrix (Failure mode analysis)
    show_confusion_matrix(nb_data[3], y_pred_nb, "Naive Bayes Confusion Matrix")
    # Visualize probability map
    visualize_naive_bayes_probs(nb_cond_probs)



    # Linear Classifier
    y_pred_linear, W = linear_classifier(*linear_data)
    # Show confusion matrix (Failure mode analysis)
    show_confusion_matrix(linear_data[3], y_pred_linear, "Linear Classifier Confusion Matrix")
    # Show Linear weights
    visualize_linear_weights(W)



    # MLP
    y_pred_mlp = mlp_classifier(*mlp_data)
    # Show confusion matrix (Failure mode analysis)
    show_confusion_matrix(mlp_data[3], y_pred_mlp, "MLP Confusion Matrix")


    #CNN
    y_pred_cnn = cnn_classifier(*cnn_data)
    # Show confusion matrix (Failure mode analysis)
    show_confusion_matrix(cnn_data[3], y_pred_cnn, "CNN Confusion Matrix")


if __name__ == "__main__":
    main()
