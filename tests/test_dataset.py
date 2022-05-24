"""Test Module for dataset."""
from pathlib import Path

import torch
from torchvision import transforms

from boxsup_pytorch.data.dataset import BoxSupDataset


class TestBoxSupDataset():

    def test_dataset_genetate_maskfrom_xml(self):
        root = Path(__file__).parent / "data/train"

        dataset = BoxSupDataset(root)

        mask = dataset._generate_mask_from_xml(dataset.label_list[0])
        assert mask.shape == torch.Size([10, 1024, 1024])
        assert mask[0].max() == 6
        assert mask[1].max() == 1
        assert mask[2].max() == 1
        assert mask[3].max() == 3
        assert mask[4].max() == 3
        assert mask[5].max() == 3
        assert mask[6].max() == 3
        assert mask[7].max() == 2
        assert mask[8].max() == 4
        assert mask[9].max() == 5

    def test_dataset_get(self):
        root = Path(__file__).parent / "data/train"

        dataset = BoxSupDataset(root)

        sample = dataset[0]
        assert sample[0].size == (1024, 1024)
        assert sample[1].shape == torch.Size([10, 1024, 1024])

    def test_dataset_get_with_transforms(self):
        root = Path(__file__).parent / "data/train"

        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.20285773090479237, 0.20285773090479237, 0.20285773090479237],
                    [0.11217139978863681, 0.11217139978863681, 0.11217139978863681]
                )
            ])
        target_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

        dataset = BoxSupDataset(root, transform, target_transform)

        sample = dataset[0]
        assert sample[0].shape == torch.Size([3, 224, 224])
        assert sample[1].shape == torch.Size([10, 224, 224])

    def test_dataset_len(self):
        root = Path(__file__).parent / "data/train"

        dataset = BoxSupDataset(root)

        length = len(dataset)
        assert length == 1
