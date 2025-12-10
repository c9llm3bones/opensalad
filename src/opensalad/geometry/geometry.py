"""PyTorch geometry utilities for protein structures.

Adapted from OpenFold (https://github.com/aqlaboratory/openfold)
Provides Vec3Array and rotation utilities needed for structure manipulation.
"""

from __future__ import annotations

import dataclasses
from typing import Union, Tuple

import torch

Float = Union[float, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Vec3Array:
    """Immutable 3D vector class for geometry operations."""
    x: torch.Tensor = dataclasses.field(metadata={'dtype': torch.float32})
    y: torch.Tensor
    z: torch.Tensor

    def __post_init__(self):
        if hasattr(self.x, 'dtype'):
            assert self.x.dtype == self.y.dtype, f"dtype mismatch: {self.x.dtype} vs {self.y.dtype}"
            assert self.x.dtype == self.z.dtype, f"dtype mismatch: {self.x.dtype} vs {self.z.dtype}"
            assert all([x == y for x, y in zip(self.x.shape, self.y.shape)]), "shape mismatch"
            assert all([x == z for x, z in zip(self.x.shape, self.z.shape)]), "shape mismatch"

    def __add__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x * other,
            self.y * other,
            self.z * other,
        )

    def __rmul__(self, other: Float) -> Vec3Array:
        return self * other

    def __truediv__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x / other,
            self.y / other,
            self.z / other,
        )

    def __neg__(self) -> Vec3Array:
        return self * -1

    def __pos__(self) -> Vec3Array:
        return self * 1

    def __getitem__(self, index) -> Vec3Array:
        return Vec3Array(
            self.x[index],
            self.y[index],
            self.z[index],
        )

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    @property
    def shape(self):
        return self.x.shape

    def map_tensor_fn(self, fn) -> Vec3Array:
        """Apply a function to each component."""
        return Vec3Array(
            fn(self.x),
            fn(self.y),
            fn(self.z),
        )

    def cross(self, other: Vec3Array) -> Vec3Array:
        """Compute cross product."""
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return Vec3Array(new_x, new_y, new_z)

    def norm(self, **kwargs) -> torch.Tensor:
        """Compute L2 norm."""
        return torch.sqrt(
            self.x ** 2 + self.y ** 2 + self.z ** 2 + 1e-8
        )

    def norm2(self) -> torch.Tensor:
        """Compute squared L2 norm."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def to_array(self) -> torch.Tensor:
        """Convert to (*, 3) tensor."""
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @staticmethod
    def from_array(arr: torch.Tensor) -> Vec3Array:
        """Create from (*, 3) tensor."""
        *batch, three = arr.shape
        assert three == 3
        x, y, z = torch.unbind(arr, dim=-1)
        return Vec3Array(x, y, z)


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply two rotation matrices [*, 3, 3]."""
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [row_mul(0), row_mul(1), row_mul(2)],
        dim=-2
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrix r to vector t: r @ t."""
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def batch_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise distances between points in a and b.
    
    Args:
        a: (batch, n, 3)
        b: (batch, m, 3)
    Returns:
        (batch, n, m) pairwise distances
    """
    diff = a.unsqueeze(2) - b.unsqueeze(1)  # (batch, n, m, 3)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise distances (non-batched).
    
    Args:
        a: (n, 3)
        b: (m, 3)
    Returns:
        (n, m) distances
    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)  # (n, m, 3)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def extract_aa_frames(
    positions: Vec3Array
) -> Tuple[torch.Tensor, Vec3Array]:
    """Extract local frames from C-alpha positions.
    
    From 4 atoms (N, CA, C, CB), extract:
    - rotation matrix (3, 3)
    - relative positions w.r.t. CA
    
    Args:
        positions: Vec3Array of shape (N, 4, 3) for N,CA,C,CB atoms
    Returns:
        rotation_matrix: (N, 3, 3)
        local_positions: Vec3Array of local frame positions
    """
    # This is a simplified version. Full implementation would:
    # 1. Compute rotation matrix from 4 atoms
    # 2. Return local coordinates in that frame
    # For now, return identity rotation and positions
    batch_size = positions.x.shape[0]
    device = positions.x.device
    dtype = positions.x.dtype
    
    rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    local_pos = positions
    
    return rot, local_pos
