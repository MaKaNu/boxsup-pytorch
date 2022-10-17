"""BoxSup Dataset Module."""
import json
import logging
from pathlib import Path
import pickle
from typing import Dict

from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset

LOGGER_NAME = __name__.split('.')[-1]  # Get Module name without package
LOGGER = logging.getLogger(LOGGER_NAME)


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

    def _convert_bbox_to_idx(self, bbox_tensor):
        classes = bbox_tensor.amax(axis=(1, 2))
        flatten_bbox = bbox_tensor.flatten(start_dim=1)
        bbox_list = []
        assert classes.shape[0] == flatten_bbox.shape[0]
        for idx, bbox in enumerate(flatten_bbox):
            bbox_list.append((int(classes[idx]), bbox.nonzero().flatten()))
        return bbox_list


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
        self.image_list = sorted(list(root_dir.glob('*.png')))
        self.bbox_list = sorted(list(root_dir.glob('*.xml')))
        self.mask_list = sorted(list(root_dir.glob('*.pkl')))
        assert len(self.image_list) > 0, "Dataset is incomplete, missing images."
        assert len(self.bbox_list) > 0, "Dataset is incomplete, missing bboxes."
        assert len(self.mask_list) > 0, "Dataset is incomplete, missing masks."
        assert len(self.image_list) == len(self.bbox_list) == len(self.mask_list)
        with open(root_dir.parent / 'classes.json', 'r') as file:
            dict_str = file.read()
        self.classes_dict = json.loads(dict_str)

    def __getitem__(self, index: int) -> torch.Tensor:
        if torch.is_tensor(index):
            index = index.tolist()

        # Get Image Data
        img_file = self.image_list[index]
        loaded_img = Image.open(img_file)
        img = self._i32_to_rgb(loaded_img)

        # Get BBox Data
        bbox_file = self.bbox_list[index]
        bbox = self._generate_mask_from_xml(bbox_file)

        # Get Mask Data
        mask_file = self.mask_list[index]
        with open(mask_file, 'rb') as file:
            mask = pickle.load(file)

        LOGGER.info(f"Loaded Image {img_file}")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            bbox = self.target_transform(bbox)
        bbox = self._convert_bbox_to_idx(bbox)

        return img, bbox, mask, img_file


class BoxSupUpdateDataset(BoxSupBaseDataset):
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
