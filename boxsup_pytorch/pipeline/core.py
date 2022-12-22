"""Pipeline core module."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from PIL.Image import Image
from torch import Tensor

GLOBAL_PIPELINE_CONFIG: List[PipelineProcess | Pipeline] = [
    PreProcessMCG,
    UpdateMasks,
    UpdateNet
]


@runtime_checkable
class PipelineProcess(Protocol):
    """Process of a Pipeline."""

    def update(self):
        """Update the internal data."""
        ...  # pragma: no cover

    def set_inputs(self, inputs: Dict[str, Tensor | Image]):
        """Set Inputs of Process."""
        ...  # pragma: no cover

    def get_outputs(self) -> Dict[str, Optional[Tensor]] | None:
        """Get Outputs of Process."""
        ...  # pragma: no cover


class Pipeline():
    """Pipeline class wich is able to run the PipelineProcesses with specified input."""

    def __init__(self, input,
                 config: List[PipelineProcess | Pipeline] = GLOBAL_PIPELINE_CONFIG) -> None:
        """Initialze Class Pipeline.

        Args:
            input (_type_): _description_
            config (List[PipelineProcess], optional): List of PipelineProcess Instances.
                Defaults to GLOBAL_PIPELINE_CONFIG.
        """
        self.config = config
        self._get_process_inputs(config[0])
        self.input = input

    def run(self):
        """Run the Pipeline itarative."""
        pass

    def _get_process_inputs(self, process: PipelineProcess | Pipeline):
        inputs = [attr for attr in process.__dict__.keys() if attr.startswith('in_')]
        self.input_names = inputs
