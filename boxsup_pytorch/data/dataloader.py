"""Dataloader Module."""
import copy
import logging
from pathlib import Path
import time

import torch
from torch.backends import cudnn
from torchvision import transforms
import yaml

from boxsup_pytorch.config.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataset import BoxSupDataset

# Global Logger
LOGGER_NAME = __name__.split('.')[0]
LOGGER = logging.getLogger(LOGGER_NAME)


class BoxSupDataloader():
    def __init__(self) -> None:
        dataset = GLOBAL_CONFIG.config().get(section="DEFAULT", option="dataset")
        self.data_dir = Path(__file__).resolve().parent / 'datasets' / dataset

        # Load mean and std of dataset
        with open(self.data_dir / 'mean_std.yml', "r") as file:
            statistic_data = yaml.safe_load(file)
        mean = statistic_data['mean']
        std = statistic_data['std']
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
        ])
        target_transfrom = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

        self.image_datasets = {
            x: BoxSupDataset(
                self.data_dir / x,
                transform,
                target_transfrom
            ) for x in ['train', 'val']
        }

    def get_data_loader(self):
        batch_size = GLOBAL_CONFIG.config().getany(
            section="DEFAULT", option="batchsize", fallback=20
        )

        dataloaders = {x: torch.utils.data.DataLoader(
            self.image_datasets[x],
            batch_size=batch_size,
            shuffle=True) for x in ['train', 'val']
        }
        return dataloaders

    def get_dataset_sizes(self):
        return {x: len(self.image_datasets[x]) for x in ['train', 'val']}

    def get_class_names(self):
        return self.image_datasets['train'].classes_dict


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=80):
    """Trains a given model.

    Args:
        model (_type_): Network Model
        dataloaders (_type_): Dataset Dataloader
        criterion (_type_): Loss Criterion
        optimizer (_type_): Optimizer Object
        scheduler (_type_): Learningrate Scheduler
        num_epochs (int, optional): Number of repeated Training iterations. Defaults to 80.

    Returns:
        _type_: trained model
    """
    since = time.time()

    device = _device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        x: GLOBAL_CONFIG.config().getany(section="DEFAULT", option=f"num_{x}") for x in [
            'train', 'val'
        ]
    }        # Decay LR by a factor of 0.1 every 7 epochs

    for epoch in range(num_epochs):
        LOGGER.info("-"*10)
        LOGGER.info(f"Epoch {epoch}/{num_epochs - 1}")
        LOGGER.info("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward Propagation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[0], 1)
                    loss = criterion(outputs[0], labels.type(torch.long))

                    # Backward Propagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            LOGGER.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        LOGGER.info("")

    time_elapsed = time.time() - since
    LOGGER.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    LOGGER.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _device():
    if torch.cuda.is_available():
        cudnn.benchmark = True
        use_device = "cuda"
    else:
        use_device = "cpu"
    return torch.device(use_device)
