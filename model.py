"""Model definitions shared by the training notebook and predict.py.

Keeping the architectures here means the saved weights (models/cnn_mnist.pt)
can be loaded into an identical, freshly built network anywhere.
"""
import torch.nn as nn


def build_cnn():
    """CNN: 2 conv+pool blocks followed by a dense head. Input [N, 1, 28, 28] -> 10 scores."""
    return nn.Sequential(
        # Block 1: 1 -> 32 feature maps, 28x28 -> 14x14
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 2: 32 -> 64 feature maps, 14x14 -> 7x7
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Dense head: 64*7*7 = 3136 -> 128 -> 10
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def build_mlp():
    """Plain fully-connected baseline. Flattens the image first, so pixel neighborhoods are lost."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
