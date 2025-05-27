"""
    Provides custom dataset classes. Currently includes:
    - NoiseDataset: Used to generate batches of random Gaussian noise.
"""

import torch

class NoiseDataset:
    """Class for generating batches of Gaussian noise vectors."""

    def __init__(self, dim=2):
        self.dim = dim

    def next_batch(self, batch_size=64):
        """Returns a batch of random noise vectors."""
        return torch.randn(batch_size, self.dim)