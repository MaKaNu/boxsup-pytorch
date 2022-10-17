"""Pipeline core module."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PipelineProcess(Protocol):
    """Process of a Pipeline."""

    def update(self):
        """Update the internal data."""
        ...  # pragma: no cover

    def set_inputs(self, **kwargs):
        """Set Inputs of Process."""
        ...  # pragma: no cover

    def get_outputs(self):
        """Get Outputs of Process."""
        ...  # pragma: no cover


def create_mask_from_index(mask_indices: torch.Tensor, mask_class: int, resolution) -> torch.Tensor:
    """Create Class Mask from Index and Class.

    Args:
        mask_indices (torch.Tensor): Tensor of indices which represents the class
        mask_class (int): Class Number
        resolution (torch.Size): Resolution of Mask

    Returns:
        torch.Tensor: The Created Mask
    """
    created_mask = torch.zeros(resolution).to(torch.int64)
    candidate_mask_class = torch.Tensor([mask_class]).to(torch.int64)
    candidate_mask_indices = mask_indices.to(torch.int64)
    created_mask.flatten()[candidate_mask_indices] = candidate_mask_class
    return created_mask
