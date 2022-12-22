"""Process Greedy Start Module."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, final, Optional

from PIL.Image import Image
import torch
from torch import Tensor
import torch.nn.functional as F

from boxsup_pytorch.pipeline.error_calc import ErrorCalc
from boxsup_pytorch.utils.check import check_exists_msg, check_init_msg

@dataclass
class BaseStrat(ABC):

    error_calc: ErrorCalc
    in_image: Optional[Image] = None
    in_masks: Optional[Tensor] = None
    in_bboxes: Optional[Tensor] = None
    out_labelmask: Optional[Tensor] = None

    @abstractmethod
    def update(self):
        ...

    @final
    def set_inputs(self, inputs: Dict[str, Image | Tensor]):
        assert "image" in inputs.keys(), check_exists_msg("image")
        assert "bboxes" in inputs.keys(), check_exists_msg("bboxes")
        assert "masks" in inputs.keys(), check_exists_msg("masks")

        self.in_image = inputs['image'] if isinstance(inputs['image'], Image) else None
        self.in_bboxes = inputs['bboxes'] if isinstance(inputs['bboxes'], Tensor) else None
        self.in_masks = inputs['masks'] if isinstance(inputs['masks'], Tensor) else None

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
        assert self.in_bboxes is not None, check_init_msg()
        assert self.in_masks is not None, check_init_msg()
        
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
