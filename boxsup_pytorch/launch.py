"""Main module."""
import logging
import logging.config
import logging.handlers
from pathlib import Path

import torch
from torch.backends import cudnn
from torchvision import transforms
import yaml

logging_config_file = Path(__file__).parent.parent.resolve() / 'logging.conf'
logging.config.fileConfig(logging_config_file)

from config.config import GLOBAL_CONFIG
from data.dataset import BoxSupDataset
from model.network import FCN8s
from pipeline.error_calc import ErrorCalc
from pipeline.strats import GreedyStrat, MiniMaxStrat
from pipeline.update_masks import UpdateMasks
from pipeline.update_net import UpdateNetwork


def main():
    """Call the main routine."""
    logger = logging.getLogger(__name__)

    # load config data
    nclass = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="num_classes")
    dataset_name = GLOBAL_CONFIG.config().get(section="DEFAULT", option="dataset")
    num_iter = GLOBAL_CONFIG.config().getany(section="DEFAULT", option="num_iter")

    data_dir = Path(__file__).parent.resolve() / f"data/datasets/{dataset_name}"

    # Specify the device the network is running on
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open(data_dir / 'mean_std.yml', "r") as file:
        statistic_data = yaml.safe_load(file)
    mean = statistic_data['mean']
    std = statistic_data['std']
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
    ])

    target_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    dataset = BoxSupDataset(
        data_dir / 'val',
        transform,
        target_transform
    )

    # Prepare Network
    network = FCN8s(nclass=nclass, device=device).to(device)

    # Load Processes
    error_calc = ErrorCalc(network)
    greedy = GreedyStrat(error_calc)
    minimax = MiniMaxStrat(error_calc)
    update_masks = UpdateMasks(greedy, minimax, dataset)
    update_net = UpdateNetwork(network)

    logger.info("Begin Training")
    for i in range(num_iter):
        logger.info(f"Iteration {i} started")
        update_masks.update()
        images_and_masks = update_masks.get_outputs()
        update_net.set_inputs(**images_and_masks)
        update_net.update()


if __name__ == "__main__":
    main()  # pragma: no cover
