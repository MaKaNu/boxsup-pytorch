"""ErrorCalc Pipeline Module."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

from ..awe_sem_seg.core.models.fcn import get_fcn8s
from ..losses import overlapping_loss, regression_loss, weighted_loss


@dataclass
class ErrorCalc:
    """The ErrorCalc Process."""

    in_image: Image = None
    in_masks: npt.NDArray[np.float64] = None
    in_bbox: npt.NDArray[np.float64] = None
    out_masks: npt.NDArray[np.float64] = None
    out_loss: Union[np.float64, npt.NDArray[np.float64]] = None

    def update(self):
        """Update the ErrorCalc Process."""
        predicted_classes = self._network_inference()
        overlap_e = overlapping_loss(self.in_masks, self.in_bbox)
        regression_e = regression_loss(predicted_classes, self.in_masks)
        self.out_masks = self.in_masks
        self.out_loss = weighted_loss(overlap_e, regression_e)

    def _network_inference(self):
        """Calculate network inference.

        Returns:
            np.array: Array with Class predictions
        """

        # Setup Model
        model = get_fcn8s(
            dataset='mars',
            backbone='vgg16',
            pretrained=True,
            local_rank=0
        )
        if torch.cuda.is_available():
            cudnn.benchmark = True
            use_device = "cuda"
        else:
            use_device = "cpu"
        device = torch.device(use_device)
        model.to(device)
        model.eval()
        
        # Begin Inference
        image = image.to(self.device)
        with torch.no_grad():
                outputs = model(self.in_image)
        return outputs
