"""Utility functions for geometric indexing and aggregation.

Adapted from SALAD and OpenFold.
"""

import torch
from typing import Optional


def index_mean(
    data: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """Compute per-index mean of data.
    
    Args:
        data: (N, ...) to aggregate
        index: (N,) integer indices
        mask: (N,) or (N, 1) boolean mask
    Returns:
        (max_index+1, ...) per-index means
    """
    if mask is None:
        mask = torch.ones(data.shape[0], dtype=torch.bool, device=data.device)
    
    # Ensure mask has compatible shape
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    
    batch_size = index.max().item() + 1
    
    # Masked sum
    masked_data = torch.where(mask, data, torch.zeros_like(data))
    sum_val = torch.zeros(batch_size, *data.shape[1:], 
                          dtype=data.dtype, device=data.device)
    sum_val = sum_val.scatter_add_(0, index.unsqueeze(-1).expand_as(masked_data), masked_data)
    
    # Count per index
    count = torch.zeros(batch_size, dtype=data.dtype, device=data.device)
    count = count.scatter_add_(0, index, mask.float().squeeze(-1))
    
    return sum_val / torch.clamp(count.unsqueeze(-1), min=1e-8)


def index_sum(
    data: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """Compute per-index sum of data.
    
    Args:
        data: (N,) to aggregate
        index: (N,) integer indices
        mask: (N,) boolean mask
    Returns:
        (max_index+1,) per-index sums
    """
    if mask is None:
        mask = torch.ones(data.shape[0], dtype=torch.bool, device=data.device)
    
    batch_size = index.max().item() + 1
    masked_data = torch.where(mask, data, torch.zeros_like(data))
    
    sum_val = torch.zeros(batch_size, dtype=data.dtype, device=data.device)
    sum_val = sum_val.scatter_add_(0, index, masked_data)
    
    return sum_val


def distance_rbf(
    distances: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 22.0,
    n_bins: int = 16
) -> torch.Tensor:
    """Radial basis function encoding of distances.
    
    Args:
        distances: distance values
        d_min, d_max: range
        n_bins: number of RBF bins
    Returns:
        (*, n_bins) RBF features
    """
    # Create RBF centers
    centers = torch.linspace(d_min, d_max, n_bins, device=distances.device, dtype=distances.dtype)
    sigma = (d_max - d_min) / (n_bins - 1)
    
    # Compute RBF
    diff = distances.unsqueeze(-1) - centers
    rbf = torch.exp(-0.5 * (diff / sigma) ** 2)
    
    return rbf


def distance_one_hot(
    distances: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 22.0,
    n_bins: int = 16
) -> torch.Tensor:
    """One-hot encoding of distances into bins.
    
    Args:
        distances: distance values
        d_min, d_max: range
        n_bins: number of bins
    Returns:
        (*, n_bins) one-hot encoded features
    """
    bin_size = (d_max - d_min) / n_bins
    bin_indices = ((distances - d_min) / bin_size).clamp(0, n_bins - 1).long()
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=n_bins).float()
    return one_hot


def get_neighbours(
    distances: torch.Tensor,
    mask: torch.Tensor,
    k: int = 16
) -> torch.Tensor:
    """Select k nearest neighbours by distance.
    
    Args:
        distances: (N, N) distance matrix
        mask: (N, N) boolean mask of valid pairs
        k: number of neighbours
    Returns:
        (N, k) indices of k nearest neighbours
    """
    # Set invalid distances to inf
    distances = torch.where(mask, distances, torch.full_like(distances, float('inf')))
    
    # Get k smallest
    _, neighbours = torch.topk(distances, k, dim=1, largest=False)
    
    return neighbours


class Vec3:
    """Minimal Vec3 helper for 3D positions."""
    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, dim=-1)

    @staticmethod
    def pairwise_dist(x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3)
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        return torch.sqrt((diff ** 2).sum(-1) + 1e-8)
