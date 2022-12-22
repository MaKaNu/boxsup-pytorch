"""ErrorCalc Pipeline Module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PIL.Image import Image
import torch
from torch import Tensor
from torchvision import transforms  # type: ignore

from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.utils.check import check_exists_msg, check_init_msg
from boxsup_pytorch.utils.losses import Losses


@dataclass
class ErrorCalc:
    """The ErrorCalculation Process."""

    network: FCN8s
    losses: Losses
    in_image: Optional[Image] = None
    in_masks: Optional[Tensor] = None
    in_bbox: Optional[Tensor] = None
    out_loss: Optional[Tensor] = None

    def update(self):
        """Update the ErrorCalc Process."""
        assert self.in_bbox is not None, check_init_msg()
        assert self.in_masks is not None, check_init_msg()

        network_output = self._network_inference()
        overlap_e = self.losses.overlapping_loss(self.in_bbox, self.in_masks)
        regression_e = self.losses.regression_loss(network_output, self.in_masks)
        self.out_loss = self.losses.weighted_loss(overlap_e, regression_e)

    def set_inputs(self, inputs: Dict[str, Tensor | Image]) -> None:
        """Set values for the ErrorCalc process inputs.

        Args:
            inputs (Dict[str, Tensor  |  Image]): dict which holds the values with
                                                  specific name as key.
        """
        assert "image" in inputs.keys(), check_exists_msg("image")
        assert "bbox" in inputs.keys(), check_exists_msg("bbox")
        assert "masks" in inputs.keys(), check_exists_msg("masks")

        self.in_image = inputs['image'] if isinstance(inputs['image'], Image) else None
        self.in_bbox = inputs['bbox'] if isinstance(inputs['bbox'], Tensor) else None
        self.in_masks = inputs['masks'] if isinstance(inputs['masks'], Tensor) else None

    def get_outputs(self) -> Dict[str, Optional[Tensor]]:
        """Get values of the ErrorCalc process outputs.

        Returns:
            Dict[str, Optional[Tensor]]: Returns the calculated loss
        """
        return {'loss': self.out_loss}

    def _network_inference(self) -> Tensor:
        # Setup Model
        self.network.eval()

        # Begin Inference
        trans = transforms.ToTensor()
        image = trans(self.in_image).to(self.network.device)
        image = image[None, :]  # Add Dummy Batch Dim
        with torch.no_grad():
            outputs = self.network(image)
        return outputs[0].cpu()
