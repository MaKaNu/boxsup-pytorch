"""ErrorCalc Pipeline Module."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image

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
        # TODO realize real function
        return np.random.rand(1, 5, 25, 25)
