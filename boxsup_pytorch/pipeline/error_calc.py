"""ErrorCalc Pipeline Module."""

from dataclasses import dataclass
from typing import Optional

from PIL import Image
import torch
from torchvision import transforms

from boxsup_pytorch.model.network import FCN8s
from ..losses import overlapping_loss, regression_loss, weighted_loss


@dataclass
class ErrorCalc:
    """The ErrorCalc Process."""

    network: FCN8s
    in_image: Optional[Image.Image] = None
    in_masks: Optional[torch.Tensor] = None
    in_bbox: Optional[torch.Tensor] = None
    out_loss: Optional[torch.Tensor] = None

    def update(self):
        """Update the ErrorCalc Process."""
        network_output = self._network_inference()[0].cpu()
        overlap_e = overlapping_loss(self.in_bbox, self.in_masks)
        regression_e = regression_loss(network_output, self.in_masks)
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
        trans = transforms.ToTensor()
        image = trans(self.in_image).to(self.network.device)
        image = image[None, :]  # Add Dummy Batch Dim
        with torch.no_grad():
            outputs = self.network(image)
        return outputs
