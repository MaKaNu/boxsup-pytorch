"""Dataloader Module."""
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.core.dataset_factory import dataset_factory


class BoxSupDataloader:
    """BoxSup initiate training class."""

    def __init__(self) -> None:
        self.dataset_root = GLOBAL_CONFIG.root
        dataset_mean_file = self.dataset_root / "ImageSets/BoxSup/mean_std.yml"

        # Load mean and std of dataset
        with open(dataset_mean_file, "r") as file:
            statistic_data = yaml.safe_load(file)
        mean = statistic_data["mean"]
        std = statistic_data["std"]
        self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
        )
        self.target_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224)
                ]
            )

        self.train_dataset = dataset_factory.get_dataset("ALL")(
            self.dataset_root, type="train",
            transform=self.transform, target_transform=self.target_transform
        )
        self.val_dataset = dataset_factory.get_dataset("ALL")(
            self.dataset_root, type="val",
            transform=self.transform, target_transform=self.target_transform
        )

    def get_data_loader(self):
        batch_size = GLOBAL_CONFIG.batchsize

        dataloaders = {
            x.type: DataLoader(x, batch_size=batch_size, shuffle=True)
            for x in [self.train_dataset, self.val_dataset]
        }
        return dataloaders

    def get_dataset_sizes(self):
        return {x.type: len(x) for x in [self.train_dataset, self.val_dataset]}

    def get_class_names(self):
        return {x.type: x.classes for x in [self.train_dataset, self.val_dataset]}


if __name__ == "__main__":
    loader = BoxSupDataloader()

    print(loader.get_dataset_sizes())
    print(loader.get_class_names())
    dataloaders = loader.get_data_loader()

    print(dataloaders)
