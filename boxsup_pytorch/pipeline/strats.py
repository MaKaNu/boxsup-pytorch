"""Process Greedy Start Module."""
from dataclasses import dataclass
import logging
from typing import List, Optional, Union

from PIL import Image
import torch
import torch.nn.functional as F

from boxsup_pytorch.pipeline.core import create_mask_from_index
from boxsup_pytorch.pipeline.error_calc import ErrorCalc

LOGGER_NAME = __name__.split('.')[-1]
LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class BaseStrat():

    error_calc: ErrorCalc
    in_image: Optional[Image.Image] = None
    in_masks: Optional[List[Union[int, torch.Tensor]]] = None
    in_bboxes: Optional[List[Union[int, torch.Tensor]]] = None
    out_labelmask: Optional[torch.Tensor] = None

    def update(self):
        pass

    def set_inputs(self, **kwargs):
        assert "image" in kwargs.keys()
        assert "bboxes" in kwargs.keys()
        assert "masks" in kwargs.keys()

        self.in_image = kwargs['image']
        self.in_bboxes = kwargs['bboxes']
        self.in_masks = kwargs['masks']

    def get_outputs(self):
        return {'labelmask': self.out_labelmask}

    def _reduce_masks(self, stacked_masks):
        def first_nonzero(x, axis=0):
            nonz = (x > 0)
            return ((nonz.cumsum(axis) == 1) & nonz).max(axis)

        _, idx = first_nonzero(stacked_masks)
        mask = F.one_hot(idx, stacked_masks.shape[0]).permute(2, 0, 1) == 1
        target = torch.zeros_like(stacked_masks)
        target[mask] = stacked_masks[mask]
        return target.sum(dim=0)

    def __repr__(self) -> str:
        return self.__class__


@dataclass
class GreedyStrat(BaseStrat):

    def update(self):
        stacked_labelmasks = torch.zeros((len(self.in_bboxes), *self.in_image.shape[1:]))
        for idx, bbox in enumerate(self.in_bboxes):
            LOGGER.info(f"Running Greedy Strat for bbox #{idx}")
            input = {
                "image": self.in_image,
                "bbox": bbox,
                "masks": self.in_masks
            }
            self.error_calc.set_inputs(**input)
            self.error_calc.update()
            output = self.error_calc.get_outputs()
            selected_idx = self._get_min_index(output['loss'])
            selected_mask = self.in_masks[selected_idx]
            class_of_bbox = bbox[0]
            created_mask = create_mask_from_index(
                mask_indices=selected_mask[1],
                mask_class=class_of_bbox,
                resolution=self.in_image.shape[1:]
            )
            stacked_labelmasks[idx, :, :] = created_mask
        self.out_labelmask = self._reduce_masks(stacked_labelmasks)

    def __repr__(self) -> str:
        return super().__repr__()

    @staticmethod
    def _get_min_index(values):
        not_nans = ~values.isnan()
        minimas = (values == values[not_nans].min()).nonzero().squeeze()
        if minimas.shape[0] > 1:
            minimas = minimas[0]  # Select the first one (Maybe random later)
        return minimas


@dataclass
class MiniMaxStrat(BaseStrat):

    def update(self):
        stacked_labelmasks = torch.zeros((len(self.in_bboxes), *self.in_image.shape[1:]))
        for idx, bbox in enumerate(self.in_bboxes):
            LOGGER.info(f"Running MiniMax Strat for bbox #{idx}")
            input = {
                "image": self.in_image,
                "bbox": bbox,
                "masks": self.in_masks
            }
            self.error_calc.set_inputs(**input)
            self.error_calc.update()
            output = self.error_calc.get_outputs()
            topk_idx = self._get_topk_index(output['loss'], 5)
            random_idx = torch.randint(topk_idx.shape[0], (1,))
            selected_idx = topk_idx[random_idx]
            selected_mask = self.in_masks[selected_idx]
            class_of_bbox = bbox[0]
            created_mask = create_mask_from_index(
                mask_indices=selected_mask[1],
                mask_class=class_of_bbox,
                resolution=self.in_image.shape[1:]
            )
            stacked_labelmasks[idx, :, :] = created_mask
        self.out_labelmask = self._reduce_masks(stacked_labelmasks)

    def __repr__(self) -> str:
        return super().__repr__()

    @staticmethod
    def _get_topk_index(values, k):
        values = values.nan_to_num(nan=0)
        _, topk_indices = values.topk(5)
        return topk_indices
