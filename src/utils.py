"""
Preprocessing:
- Normalize pixel values to [0,1] (NumPy) or [-1,1] (PyTorch with transforms.Normalize).
- Flatten into vectors (784 features) when needed.
- Keep 2D shape for CNNs.
"""