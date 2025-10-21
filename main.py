"""

"""
from src.utils import prepare_all_versions
from src.knn import knn_classifier
from src.naive_bayes import naive_bayes_classifier
from src.linear_classifier import linear_classifier
from src.mlp import mlp_classifier
from src.cnn import cnn_classifier


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
    knn_classifier(*knn_data, k_values=[1,2,3])
    naive_bayes_classifier(*nb_data)
    linear_classifier(*linear_data)
    mlp_classifier(*mlp_data)
    cnn_classifier(*cnn_data)
    

if __name__ == "__main__":
    main()
