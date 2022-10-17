"""Test Module for dataloader."""

from torch.utils.data.dataloader import DataLoader

from boxsup_pytorch.data.dataloader import BoxSupDataloader


class TestBoxSupDataloader:

    def test_dataloader_creation(self):
        dataloader = BoxSupDataloader()

        assert 'train', 'val' in dataloader.image_datasets.keys()

    def test_dataloader_get_datasets(self):
        dataloader = BoxSupDataloader()

        dataloaders = dataloader.get_data_loader()

        assert 'train', 'val' in dataloaders.keys()
        assert isinstance(dataloaders['train'], DataLoader)
        assert isinstance(dataloaders['val'], DataLoader)
