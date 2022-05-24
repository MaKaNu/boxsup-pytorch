"""Process Greedy Start Module."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from boxsup_pytorch.config.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataloader import train_model
from boxsup_pytorch.data.dataset import BoxSupUpdateDataset
from boxsup_pytorch.model.network import FCN8s


@dataclass
class UpdateNetwork():

    network: FCN8s
    in_images: Optional[torch.Tensor] = None
    in_masks: Optional[torch.Tensor] = None
    out_labelmasks: Optional[torch.Tensor] = None

    def update(self):
        num_train = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="num_train")
        num_val = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="num_val")
        
        data_train = {
            'img': self.in_images[0:num_train],
            'lbl': self.in_masks[0:num_train]
        }

        data_val = {
            'img': self.in_images[num_train:num_train + num_val],
            'lbl': self.in_masks[num_train:num_train + num_val]
        }

        dataset_train = BoxSupUpdateDataset(data_train)
        dataset_val = BoxSupUpdateDataset(data_val)

        # Load Config Data
        batch_size = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="batch_size")
        num_workers = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="num_workers")

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        dataloader_val = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        dataloaders = {
            'train': dataloader_train,
            'val': dataloader_val
        }

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        train_model(self.network, dataloaders, criterion, optimizer, scheduler, num_epochs=80)

    def set_inputs(self, **kwargs):
        assert "images" in kwargs.keys()
        assert "masks" in kwargs.keys()

        self.in_images = kwargs['images']
        self.in_masks = kwargs['masks']

    def get_outputs(self):
        return {'labelmasks': self.out_labelmasks}
