"""Process Greedy Start Module."""
from dataclasses import dataclass, field
import logging
from typing import List, Optional

import torch

from boxsup_pytorch.config.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataset import BoxSupDataset
from boxsup_pytorch.pipeline.strats import GreedyStrat, MiniMaxStrat


# Get Global Logger
LOGGER_NAME = __name__.split('.')[-1]
LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass
class UpdateMasks():

    greedy: GreedyStrat
    minimax: MiniMaxStrat
    dataset: BoxSupDataset
    out_labelmasks: Optional[List[torch.Tensor]] = field(default_factory=lambda: [])
    out_images: Optional[List[torch.Tensor]] = field(default_factory=lambda: [])

    def update(self):
        for image, bboxes, masks in self.dataset:
            LOGGER.info(f"Update Masks for image with shape {image.shape}")
            input_values = {
                'image': image,
                'bboxes': bboxes,
                'masks': masks
            }
            strat = GLOBAL_CONFIG.config().get(
                section='DEFAULT', option='strat', fallback='greedy'
            )
            if strat == 'greedy':
                self.greedy.set_inputs(**input_values)
                self.greedy.update()
                self.out_labelmasks.append(self.greedy.get_outputs()['labelmask'])
            elif strat == 'minimax':
                self.minimax.set_inputs(**input_values)
                self.minimax.update()
                self.out_labelmasks.append(self.minimax.get_outputs()['labelmask'])
            else:
                RuntimeError('Unknown strategy %s', strat)
            self.out_images.append(image)

    def get_outputs(self):
        return {
            'images': self.out_images,
            'masks': self.out_labelmasks
        }
