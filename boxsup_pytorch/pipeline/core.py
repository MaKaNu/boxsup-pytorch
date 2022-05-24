"""Pipeline core module."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


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
