"""BoxSup Dataset Module."""
import json
from pathlib import Path
from typing import Dict

from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset


class BoxSupBaseDataset(VisionDataset):
    """Base Dataset."""

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: len of image list
        """
        return len(self.image_list)

    def _i32_to_rgb(self, image):
        if image.mode == "I":
            img = np.array(image)
            new_max = 255
            img_new = (img) * ((new_max) / (img.max()))
            img_new = Image.fromarray(img_new).convert("RGB")
            return img_new
        return image

    def _generate_mask_from_xml(self, xml_file) -> torch.Tensor:
        with open(xml_file, 'r') as file:
            data = file.read()

        bs_data = BeautifulSoup(data, 'xml')

        # We need to get w/h of original Image
        width = int(bs_data.find('size').width.contents[0])
        height = int(bs_data.find('size').height.contents[0])

        # Initiate Tensor
        list_of_bboxes = bs_data.find_all('object')
        bbox_tensor = torch.zeros((
            len(list_of_bboxes),
            width,
            height
        ))
        for idx, bbox in enumerate(list_of_bboxes):
            bbox_class = self.classes_dict[bbox.find('name').contents[0]]

            # Get BBox Corners
            xmin = int(bbox.bndbox.xmin.contents[0])
            ymin = int(bbox.bndbox.ymin.contents[0])
            xmax = int(bbox.bndbox.xmax.contents[0])
            ymax = int(bbox.bndbox.ymax.contents[0])

            bbox_tensor[idx, xmin:xmax, ymin:ymax] = bbox_class
        return bbox_tensor.flip(0)  # reverse bbox order


class BoxSupDataset(BoxSupBaseDataset):
    """The BoxSupDataset Class."""

    def __init__(self, root_dir: Path, transform=None, target_transform=None):
        """Construct an BoxSupDataset instance.

        Args:
            root_dir (Path): path to the dataset root dir.
            transform (transforms, optional): Transformations of the Dataset. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = list(root_dir.glob('*.png'))
        self.label_list = list(root_dir.glob('*.xml'))
        with open(root_dir.parent / 'classes.json', 'r') as file:
            dict_str = file.read()
        self.classes_dict = json.loads(dict_str)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(index):
            index = index.tolist()

        img_file = self.image_list[index]
        loaded_img = Image.open(img_file)
        img = self._i32_to_rgb(loaded_img)
        lbl_file = self.label_list[index]
        lbl = self._generate_mask_from_xml(lbl_file)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)

        return img, lbl


class BoxSupUpdateDataset(VisionDataset):
    """Dataset which gets updated while training."""
    def __init__(
        self, data: Dict[str, torch.Tensor], transform=None, target_transform=None
    ):
        """Construct an BoxSupDataset instance.

        Args:
            root_dir (Path): path to the dataset root dir.
            transform (transforms, optional): Transformations of the Dataset. Defaults to None.
        """
        assert 'img' in data.keys()
        assert 'lbl' in data.keys()
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Get count of images.

        Returns:
            int: Count images
        """
        return len(self.data['img'])

    def __getitem__(self, index: int) -> torch.Tensor:
        if torch.is_tensor(index):
            index = index.tolist()

        img = self.data['img'][index]
        lbl = self.data['lbl'][index]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lbl = self.target_transform(lbl)

        return img, lbl
