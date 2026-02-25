from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

class LatentModule(ABC, nn.Module):
    """
    Abstract class for latent representation learning.
    
    Args:
        input_dim: dimension of original space
        latent_dim: dimension of latent space
    """
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    @abstractmethod
    def encode(self, x):
        """Map from original space to latent space."""
        pass
    
    @abstractmethod
    def decode(self, z):
        """Map from latent space to original space."""
        pass
    
    @abstractmethod
    def loss(self, x, x_recon, **kwargs):
        """Compute loss for training."""
        pass
    
    @abstractmethod
    def fit(self, dataset, **kwargs):
        """Training loop. Returns dict with train/val losses."""
        pass
