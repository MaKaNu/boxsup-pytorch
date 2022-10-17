"""Generate python dictionary from Matlab mat files."""


from argparse import ArgumentParser
from pathlib import Path
import pickle
import random

import numpy as np
from scipy import io
import torch
from bs4 import BeautifulSoup as bs


def main():
    """Read mat-files and saves them python pickle.

    Raises:
        RuntimeError: Only raised if for some reason mutually exclusive group fails.
    """
    parser = ArgumentParser("mat2dict")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=Path, help="A Single File")
    group.add_argument("-d", "--dir", type=Path, help="A dir which includes mat-files.")
    parser.add_argument("-n", "--nclass", type=int, default=6, help="Number of classes")

    args = parser.parse_args()

    if args.file:
        if args.file.exists():
            data = io.loadmat(args.file)
            list_of_masks = _convert_data(data, args.nclass)
            pkl_file = (args.file.parent / args.file.stem).with_suffix('.pkl')
            _write_to_file(list_of_masks, pkl_file)
    elif args.dir:
        file_list = list(args.dir.glob('*.mat'))
        for file in file_list:
            if file.exists():
                pkl_file = (file.parent / file.stem).with_suffix('.pkl')
                xml_file = (file.parent / file.stem).with_suffix('.xml')
                min_area = _read_smallest_area(xml_file)
                data = io.loadmat(file)
                list_of_masks = _convert_data(data, args.nclass, min_area)
                _write_to_file(list_of_masks, pkl_file)
    else:
        raise RuntimeError("Mutually Exclusive Group failed!")


def _convert_data(data, num_classes, min_area):
    assert 'result' in data.keys(), "The saved Matlab data needs to be named 'result'!"
    assert isinstance(data['result'], np.ndarray), "Mask has to be saved from cellarray with arr"
    result = []
    if data["result"].shape[0] == 0:
        num_active_pixels = len(data["result"])
        if num_active_pixels > min_area * 0.75:
            result.append((1, torch.Tensor(data["result"].astype(np.float64))))
            return result
        else:
            return result
    for mask in data['result'][0]:
        num_active_pixels = len(mask)
        if num_active_pixels > min_area * 0.75:
            class_num = random.randint(1, num_classes)
            class_indices = torch.Tensor(mask.flatten().astype(np.float64)) - 1  # MATLAB idx
            result.append((class_num, class_indices))
    return result


def _write_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        print(f"Saved list to {file.name}")


def _read_smallest_area(xml_file):
    with open(xml_file, 'r') as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
        all_labels = bs_content.find_all("object")
        areas = _calculate_areas(all_labels)
        min_area = min(areas)
        return min_area


def _calculate_areas(list_of_labels):
    areas = []
    for label in list_of_labels:
        xmin = int(label.xmin.text)
        xmax = int(label.xmax.text)
        ymin = int(label.ymin.text)
        ymax = int(label.ymax.text)
        areas.append((xmax-xmin) * (ymax-ymin))
    return areas


if __name__ == "__main__":
    main()
