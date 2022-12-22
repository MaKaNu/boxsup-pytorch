import copy
import time

import torch
from torch.backends import cudnn
from torch.optim import lr_scheduler

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.core import optimizers
from boxsup_pytorch.data.dataloader import BoxSupDataloader
from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.utils.losses import get_segmentation_loss


class Trainer:
    def __init__(self, model) -> None:
        self.model = model
        self.optimizer = optimizers.provider.create(GLOBAL_CONFIG.optimizer)(
            model.parameters(),
            GLOBAL_CONFIG.learning_rate,
            **GLOBAL_CONFIG.hyperparams[GLOBAL_CONFIG.optimizer]
        )
        self.criterion = get_segmentation_loss()
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            **GLOBAL_CONFIG.hyperparams["StepLR"]
        )
        self.loader = BoxSupDataloader()
        self.dataloaders = self.loader.get_data_loader()

    def train_model(self):

        since = time.time()

        device = _device()
        self.model = self.model.to(device)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        dataset_sizes = {
            "train": len(self.loader.train_dataset),
            "val": len(self.loader.val_dataset),
        }  # Decay LR by a factor of 0.1 every 7 epochs

        for epoch in range(GLOBAL_CONFIG.epochs):
            print(f"Epoch {epoch}/{GLOBAL_CONFIG.epochs - 1}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # TODO: Figure out how dataloaders should react here
                for inputs, masks, bboxes, gt_label in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    bboxes = bboxes.to(device)
                    gt_label = gt_label.to(device)

                    self.optimizer.zero_grad()

                    # Forward Propagation
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                      #  _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, gt_label)

                        # Backward Propagation
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == gt_label.data)
                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:4f}")

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


if __name__ == "__main__":
    model = FCN8s(nclass=21, device=_device())
    trainer = Trainer(model)

    trainer.train_model()
