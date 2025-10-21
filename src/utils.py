"""
Preprocessing:
- Normalize pixel values to [0,1] (NumPy) or [-1,1] (PyTorch with transforms.Normalize).
- Flatten into vectors (784 features) when needed.
- Keep 2D shape for CNNs.
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_mnist(data_dir="data/MNIST"):
    """
    Load MNIST images from folders 0-9.
    Returns:
        images: np.ndarray (N,28,28), float32 [0,1]
        labels: np.ndarray (N,), int
    """
    images = []
    labels = []
    
    for label in range(10):
        folder = os.path.join(data_dir, str(label))
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                img = Image.open(path).convert("L")  # grayscale
                img_array = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Warning: could not load {path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images.")
    return images, labels

def flatten_images(images):
    """Flatten images: (N,28,28) -> (N,784)"""
    return images.reshape(images.shape[0], -1)

def binarize_images(images, threshold=0.5):
    """Binarize images: 1 if pixel > threshold else 0"""
    return (images > threshold).astype(np.int32)

def create_train_test(images, labels, test_size=0.2, random_state=42):
    """
    Split data into train/test
    test_size=0.2 -> 80% for training, 20% for testing
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)

def prepare_all_versions(data_dir="data/MNIST", test_size=0.2, random_state=42):
    """
    Load MNIST and prepare all versions needed for classifiers:
    - KNN, Linear, MLP: flattened
    - Naive Bayes: flattened + binarized
    - CNN: keep 2D + add channel
    Returns:
        dict with keys: "KNN", "NaiveBayes", "Linear", "MLP", "CNN"
        each value: tuple (X_train, X_test, y_train, y_test)
    """
    images, labels = load_mnist(data_dir)
    
    # Split
    X_train, X_test, y_train, y_test = create_train_test(images, labels, test_size, random_state)
    
    # Flattened for KNN, Linear, MLP
    X_train_flat = flatten_images(X_train)
    X_test_flat  = flatten_images(X_test)
    
    # Binarized for Naive Bayes
    X_train_bin = binarize_images(X_train_flat)
    X_test_bin  = binarize_images(X_test_flat)
    
    # 2D + channel for CNN
    X_train_cnn = X_train[:, np.newaxis, :, :]  # shape (N,1,28,28)
    X_test_cnn  = X_test[:, np.newaxis, :, :]
    
    return {
        "KNN": (X_train_flat, X_test_flat, y_train, y_test),
        "NaiveBayes": (X_train_bin, X_test_bin, y_train, y_test),
        "Linear": (X_train_flat, X_test_flat, y_train, y_test),
        "MLP": (X_train_flat, X_test_flat, y_train, y_test),
        "CNN": (X_train_cnn, X_test_cnn, y_train, y_test)
    }

# Optional: preview an image for debugging
def preview_image(image):
    """
    Print an image as ASCII
    """
    for row in image:
        print("".join(['#' if px > 0.5 else '.' for px in row]))

# Test script
if __name__ == "__main__":
    data_versions = prepare_all_versions()
    
    print("Shapes:")
    for key, (X_train, X_test, y_train, y_test) in data_versions.items():
        print(f"{key}: X_train {X_train.shape}, X_test {X_test.shape}")
    
    print("\nPreview first training image (KNN/Linear/MLP version):")
    preview_image(data_versions["KNN"][0][0])
    print("Label:", data_versions["KNN"][2][0])
