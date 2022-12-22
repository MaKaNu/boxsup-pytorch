from dataclasses import dataclass
from typing import Dict, Optional

from PIL.Image import Image
import torch
from torch import Tensor

from boxsup_pytorch.data.dataset import BoxSupDataset

from boxsup_pytorch.utils.check import check_exists_msg, check_init_msg

@dataclass
class DataBasePreparation:
    """The ErrorCalculation Process."""
    in_dataset: Optional[BoxSupDataset] = None
    out_img: Optional[Tensor] = None
    out_masks: Optional[Tensor] = None
    out_bboxes: Optional[Tensor] = None
    out_gt_label: Optional[Tensor] = None

    def update(self):
        """Update the ErrorCalc Process."""
        assert self.in_dataset is not None, check_init_msg()

    def set_inputs(self, inputs: Dict[str, BoxSupDataset]) -> None:
        """Set values for the ErrorCalc process inputs.

        Args:
            inputs (Dict[str, Tensor  |  Image]): dict which holds the values with
                                                  specific name as key.
        """
        assert "dataset" in inputs.keys(), check_exists_msg("dataset")

        self.in_dataset = inputs['dataset'] if isinstance(inputs['dataset'], BoxSupDataset) else None

    def get_outputs(self) -> Dict[str, Optional[Tensor]]:
        """Get values of the ErrorCalc process outputs.

        Returns:
            Dict[str, Optional[Tensor]]: Returns the calculated loss
        """
        return {'loss': self.out_loss}