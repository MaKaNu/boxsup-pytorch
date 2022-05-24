"""Calculates the mean and std of an image set."""
from argparse import ArgumentParser
from pathlib import Path

from alive_progress import alive_bar
import numpy as np
from PIL import Image
import yaml


def main():
    """Calculate the mean and std for image dataset."""
    parser = ArgumentParser(description="Mean and Std calculation of Image dataset")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="mars",
        help="Dataset name"
    )
    args = parser.parse_args()

    # Get Dataset
    dataset = args.dataset
    data_dir = Path(__file__).resolve().parent.parent / (
        Path('boxsup_pytorch/data/datasets') / dataset
    )
    image_list = data_dir.rglob("*.png")
    num_images = len(list(data_dir.rglob("*.png")))
    # Calculate mean and std over all images
    mean = np.array([0, 0, 0], dtype=np.float64)
    std = np.array([0, 0, 0], dtype=np.float64)
    with alive_bar() as bar:
        for image_file in image_list:
            image = Image.open(image_file)
            image = _i32_to_rgb(image)
            if isinstance(image.mean(axis=(0, 1)), np.float64):
                mean += np.array([image.mean()] * 3)
                std += np.array([image.std()] * 3)
            else:
                mean += image.mean(axis=(0, 1))
                std += image.std(axis=(0, 1))
            bar()
    mean /= num_images
    std /= num_images

    # Print Result

    # Write Data to Yaml
    data = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(data_dir / "mean_std.yml", "w") as file:
        yaml.dump(data, file)


def _i32_to_rgb(image):
    if image.mode == "I":
        img = np.array(image)
        new_max = 1
        img_new = (img) * ((new_max) / img.max())
        return img_new
    return image


if __name__ == "__main__":
    main()  # prama: no cover
