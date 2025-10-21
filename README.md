# MNIST Classification Project

This project implements several classifiers (KNN, Naive Bayes, Linear, MLP, CNN) for handwritten digit recognition using the MNIST dataset.

## Setup
Type into terminal: 
```console
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Structure
data/ - MNIST dataset 

src/ - Classifiers to Implement

``` Structure
.
└── Project1
    ├── data
    │   └── MNIST
    │       ├── 0
    │       ├── 1
    │       ├── 2
    │       ├── 3
    │       ├── 4
    │       ├── 5
    │       ├── 6
    │       ├── 7
    │       ├── 8
    │       └── 9
    ├── report
    └── src
        └── __pycache__

```

## How to Run:
Type in terminal:
```console
python main.py
```

## Sample Output:
``` Sample Output:
(.venv) Project1 % python main.py
Loaded 60000 images.

==============================
Train/Test Split Info (All Versions)
==============================

KNN Dataset:
  X_train shape: (48000, 784)
  X_test shape:  (12000, 784)
  y_train shape: (48000,)
  y_test shape:  (12000,)
  Total samples: 60000
  Train/Test ratio: 0.80 / 0.20

NaiveBayes Dataset:
  X_train shape: (48000, 784)
  X_test shape:  (12000, 784)
  y_train shape: (48000,)
  y_test shape:  (12000,)
  Total samples: 60000
  Train/Test ratio: 0.80 / 0.20

Linear Dataset:
  X_train shape: (48000, 784)
  X_test shape:  (12000, 784)
  y_train shape: (48000,)
  y_test shape:  (12000,)
  Total samples: 60000
  Train/Test ratio: 0.80 / 0.20

MLP Dataset:
  X_train shape: (48000, 784)
  X_test shape:  (12000, 784)
  y_train shape: (48000,)
  y_test shape:  (12000,)
  Total samples: 60000
  Train/Test ratio: 0.80 / 0.20

CNN Dataset:
  X_train shape: (48000, 1, 28, 28)
  X_test shape:  (12000, 1, 28, 28)
  y_train shape: (48000,)
  y_test shape:  (12000,)
  Total samples: 60000
  Train/Test ratio: 0.80 / 0.20

==============================
Running KNN Classifier (NumPy Only)
==============================
k = 1 | Accuracy = 0.9695 | Time = 45.91s
k = 2 | Accuracy = 0.9623 | Time = 41.67s
k = 3 | Accuracy = 0.9689 | Time = 41.69s
==============================


==============================
Running Naive Bayes Classifier (NumPy Only)
==============================
Naive Bayes Accuracy: 0.8337
Time taken: 0.37s
==============================


==============================
Running Linear Classifier (NumPy Only)
==============================
Epoch 1/100 | Loss: 0.1077
Epoch 20/100 | Loss: 0.0605
Epoch 40/100 | Loss: 0.0527
Epoch 60/100 | Loss: 0.0496
Epoch 80/100 | Loss: 0.0478
Epoch 100/100 | Loss: 0.0467

Linear Classifier Accuracy: 0.8208
Time taken: 7.84s
==============================


==============================
Running MLP Classifier (PyTorch)
==============================
Using device: cpu
Epoch 1/20 | Loss: 1.9060
Epoch 5/20 | Loss: 0.3331
Epoch 10/20 | Loss: 0.2442
Epoch 15/20 | Loss: 0.1903
Epoch 20/20 | Loss: 0.1535

MLP Accuracy: 0.9473
Time taken: 7.45s
==============================

==============================
Running CNN Classifier (PyTorch)
==============================
Epoch 1/5 | Loss: 0.3749
Epoch 2/5 | Loss: 0.0797
Epoch 3/5 | Loss: 0.0532
Epoch 4/5 | Loss: 0.0412
Epoch 5/5 | Loss: 0.0329

CNN Accuracy: 0.9847
Time taken: 82.53s
==============================

(.venv) Project1 % 
```



# How To Change Test and Train ratio in src/ultis.py

- go to src/ultis.py
- change two lines of code
- test_size=0.2 -> 80% for training, 20% for testing

``` Example
def create_train_test(images, labels, test_size=0.2, random_state=42):
```
AND 

```
def prepare_all_versions(data_dir="data/MNIST", test_size=0.2, random_state=42):
```