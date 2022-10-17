"""ErrorCalc Pipeline Module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from boxsup_pytorch.losses import overlapping_loss, regression_loss, weighted_loss
from boxsup_pytorch.model.network import FCN8s


@dataclass
class ErrorCalc:
    """The ErrorCalc Process."""

    network: FCN8s
    in_image: Optional[torch.Tensor] = None
    in_masks: Optional[List[Tuple[Union[int, torch.Tensor]]]] = None
    in_bbox: Optional[List[Tuple[Union[int, torch.Tensor]]]] = None
    out_loss: Optional[torch.Tensor] = None

    def update(self):
        """Update the ErrorCalc Process."""
        network_output = self._network_inference()[0].cpu()
        regression_e = regression_loss(network_output, self.in_masks)
        overlap_e = overlapping_loss(self.in_bbox, self.in_masks)
        self.out_loss = weighted_loss(overlap_e, regression_e)

    def set_inputs(self, **kwargs) -> None:
        assert "image" in kwargs.keys()
        assert "bbox" in kwargs.keys()
        assert "masks" in kwargs.keys()

        self.in_image = kwargs['image']
        self.in_bbox = kwargs['bbox']
        self.in_masks = kwargs['masks']

    def get_outputs(self):
        return {'loss': self.out_loss}

    def _network_inference(self) -> torch.Tensor:
        # Setup Model
        self.network.eval()

        # Begin Inference
        image = self.in_image.to(self.network.device)
        image = image[None, :]  # Add Dummy Batch Dim
        with torch.no_grad():
            outputs = self.network(image)
        return outputs

    def __repr__(self):
        return f"{self.__class__}"
