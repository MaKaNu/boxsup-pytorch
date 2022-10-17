from argparse import ArgumentParser
from pathlib import Path
import pickle

from alive_progress import alive_bar
import numpy as np

from boxsup_pytorch.data.dataset import BoxSupDataset
from boxsup_pytorch.losses import inter_o_union


def _write_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        print(f"Saved list to {file.name}")


def main():
    parser = ArgumentParser("Generate Dummy Data")
    parser.add_argument(
        '-p', '--path',
        type=Path,
        help="Path to imagefolder",
        default=Path("./boxsup_pytorch/data/datasets/dummy")
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['train', 'val', 'test'],
        help="Specify the phase for which the data will be created.",
        default='train'
    )
    args = parser.parse_args()

    root_dir = args.path / args.phase

    dataset = BoxSupDataset(root_dir)

    for data in dataset:
        array_of_ious = np.zeros((len(data[2]), len(data[1]) + 1))
        img_name = data[3].name
        for idx_b, bbox in enumerate(data[1]):
            with alive_bar(len(data[2]), dual_line=True, title='Calculate IoU') as bar:
                for idx_m, mask in enumerate(data[2]):
                    iou = inter_o_union(bbox[1], mask[1])
                    array_of_ious[idx_m, idx_b] = iou
                    array_of_ious[idx_m, -1] = mask[0]
                    bar.title = f'IoU for {img_name}, bbox#{idx_b:05d}'
                    bar()
        iou_file = root_dir / str(Path(img_name).stem + "_iou.pkl")
        _write_to_file(array_of_ious, iou_file)


if __name__ == "__main__":
    main()
