"""Process Greedy Start Module."""
from dataclasses import dataclass
from typing import List, Optional

import torch

from boxsup_pytorch.config.config import GLOBAL_CONFIG
from boxsup_pytorch.pipeline.strats import GreedyStrat, MiniMaxStrat


@dataclass
class UpdateMasks():

    greedy: GreedyStrat
    minimax: MiniMaxStrat
    in_images: Optional[List[torch.Tensor]] = None
    in_masks: Optional[torch.Tensor] = None
    in_bboxes: Optional[torch.Tensor] = None
    out_labelmasks: Optional[List[torch.Tensor]] = None

    def update(self):
        num_images = self.in_images.shape[0]
        self.out_labelmasks = []

        for idx in range(num_images):
            input_values = {
                'image': self.in_images[idx],
                'bboxes': self.in_bboxes[idx],
                'masks': self.in_masks
            }
            strat = GLOBAL_CONFIG.config().get(
                section='DEFAULT', option='strat', fallback='greedy'
            )
            if strat == 'greedy':
                self.greedy.set_inputs(**input_values)
                self.greedy.update()
                self.out_labelmasks.append(self.greedy.get_outputs())
            elif strat == 'minimax':
                self.minimax.set_inputs(**input_values)
                self.minimax.update()
                self.out_labelmasks.append(self.minimax.get_outputs())
            else:
                RuntimeError('Unknown strategy %s', strat)

    def set_inputs(self, **kwargs):
        assert "images" in kwargs.keys()
        assert "bboxes" in kwargs.keys()
        assert "masks" in kwargs.keys()

        self.in_images = kwargs['images']
        self.in_bboxes = kwargs['bboxes']
        self.in_masks = kwargs['masks']

    def get_outputs(self):
        return {'labelmasks': self.out_labelmasks}
