"""Process Greedy Start Module."""
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import torch
import torch.nn.functional as F

from boxsup_pytorch.pipeline.error_calc import ErrorCalc


@dataclass
class BaseStrat():

    error_calc: ErrorCalc
    in_image: Optional[Image.Image] = None
    in_masks: Optional[torch.Tensor] = None
    in_bboxes: Optional[torch.Tensor] = None
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


@dataclass
class GreedyStrat(BaseStrat):

    def update(self):
        stacked_labelmasks = torch.zeros(self.in_bboxes.shape)
        for bbox_idx in range(self.in_bboxes.shape[0]):
            input = {
                "image": self.in_image,
                "bbox": self.in_bboxes[bbox_idx],
                "masks": self.in_masks
            }
            self.error_calc.set_inputs(**input)
            self.error_calc.update()
            output = self.error_calc.get_outputs()
            selected_idx = torch.argmin(output['loss'])
            selected_mask = self.in_masks[selected_idx]
            class_of_bbox = torch.max(self.in_bboxes[bbox_idx])
            zero_mask = (selected_mask != 0)
            stacked_labelmasks[bbox_idx, zero_mask] = (
                selected_mask[zero_mask]/selected_mask[zero_mask] * class_of_bbox
            )
            self.out_labelmask = self._reduce_masks(stacked_labelmasks)


@dataclass
class MiniMaxStrat(BaseStrat):

    def update(self):
        stacked_labelmasks = torch.zeros(self.in_bboxes.shape)
        for bbox_idx in range(self.in_bboxes.shape[0]):
            input = {
                "image": self.in_image,
                "bbox": self.in_bboxes[bbox_idx],
                "masks": self.in_masks
            }
            self.error_calc.set_inputs(**input)
            self.error_calc.update()
            output = self.error_calc.get_outputs()
            _, idx = torch.topk(output['loss'], 5)
            random_idx = torch.randint(idx.shape[0], (1,))
            selected_idx = idx[random_idx]
            selected_mask = self.in_masks[selected_idx[0]]
            class_of_bbox = torch.max(self.in_bboxes[bbox_idx])
            zero_mask = (selected_mask != 0)
            stacked_labelmasks[bbox_idx, zero_mask] = (
                selected_mask[zero_mask]/selected_mask[zero_mask] * class_of_bbox
            )
            self.out_labelmask = self._reduce_masks(stacked_labelmasks)
