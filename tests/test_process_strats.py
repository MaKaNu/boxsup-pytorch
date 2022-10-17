"""Test Module for GreedyStrat Process."""
from pathlib import Path

import pytest
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from boxsup_pytorch.data.dataset import BoxSupDataset
from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.pipeline.error_calc import ErrorCalc
from boxsup_pytorch.pipeline.strats import GreedyStrat, MiniMaxStrat


@pytest.fixture()
def device():
    """Pytest fixture of device."""
    if torch.cuda.is_available():
        cudnn.benchmark = True
        use_device = "cuda"
    else:
        use_device = "cpu"
    return torch.device(use_device)


@pytest.fixture()
def error_calc(device):
    """Pytest fixture of Network."""
    net = FCN8s(nclass=7, device=device)
    net.to(device)
    process = ErrorCalc(net)
    return process


@pytest.fixture()
def input_dict():
    """Pytest Fixture of dict with image bbox and masks with size 120x120."""
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
    result = {
        'image': dataset[0][0],
        'bboxes': dataset[0][1],
        'masks': dataset[0][2][0:16]  # The First 16 Masks for testing
    }
    return result


@pytest.fixture()
def stacked_masks():
    """Pytest Fixture for Stacked Masks."""
    masks = torch.Tensor(
        [
            [
                [1, 0],
                [0, 0],
            ],
            [
                [0, 2],
                [0, 2]
            ],
            [
                [0, 1],
                [1, 0]
            ],
        ]
    ).to(torch.int64)
    return masks


class TestGreedyStrat():
    """Test the methods of the GreedyStratProcess."""

    def test_greedy_strat_set(self, error_calc, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = GreedyStrat(error_calc)

        # Run tested method
        process.set_inputs(**input_dict)

        # Asserts
        assert isinstance(process.in_image, torch.Tensor)
        assert isinstance(process.in_bboxes, list)
        assert isinstance(process.in_masks, list)
        assert process.in_image.shape == (3, 224, 224)
        assert len(process.in_bboxes) == 10
        assert isinstance(process.in_bboxes[1][0], int)
        assert isinstance(process.in_bboxes[1][1], torch.Tensor)
        assert len(process.in_masks) == 16
        assert isinstance(process.in_masks[1][0], int)
        assert isinstance(process.in_masks[1][1], torch.Tensor)

    def test_greedy_strat_get(self, error_calc, input_dict):
        """Test the get_outputs Method of Process."""
        # Prepare
        process = GreedyStrat(error_calc)
        process.set_inputs(**input_dict)

        # Run tested method
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'labelmask' in result.keys()
        assert result['labelmask'] is None

    def test_greedy_strat__reduce_masks(self, error_calc, stacked_masks):
        """Test the update _reduce_masks method of Process."""
        # Prepare
        process = GreedyStrat(error_calc)

        # Run tested method
        result = process._reduce_masks(stacked_masks)

        # Asserts
        assert result.dtype == torch.int64
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([2, 2])

    def test_greedy_strat_update(self, error_calc, input_dict):
        """Test the update method of Process."""
        # Prepare
        process = GreedyStrat(error_calc, )
        process.set_inputs(**input_dict)

        # Run tested method
        process.update()
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'labelmask' in result.keys()
        assert result['labelmask'].shape == torch.Size((224, 224))


class TestMiniMaxStrat():
    """Test the methods of the GreedyStratProcess."""

    def test_minimax_strat_set(self, error_calc, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = MiniMaxStrat(error_calc)

        # Run tested method
        process.set_inputs(**input_dict)

        # Asserts
        assert isinstance(process.in_image, torch.Tensor)
        assert isinstance(process.in_bboxes, list)
        assert isinstance(process.in_masks, list)
        assert process.in_image.shape == (3, 224, 224)
        assert len(process.in_bboxes) == 10
        assert isinstance(process.in_bboxes[1][0], int)
        assert isinstance(process.in_bboxes[1][1], torch.Tensor)
        assert len(process.in_masks) == 16
        assert isinstance(process.in_masks[1][0], int)
        assert isinstance(process.in_masks[1][1], torch.Tensor)

    def test_minimax_strat_get(self, error_calc, input_dict):
        """Test the get_outputs Method of Process."""
        # Prepare
        process = MiniMaxStrat(error_calc)
        process.set_inputs(**input_dict)

        # Run tested method
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'labelmask' in result.keys()
        assert result['labelmask'] is None

    def test_minimax_strat_update(self, error_calc, input_dict):
        """Test the update method of Process."""
        # Prepare
        process = MiniMaxStrat(error_calc)
        process.set_inputs(**input_dict)

        # Run tested method
        process.update()
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'labelmask' in result.keys()
        assert result['labelmask'].shape == torch.Size((224, 224))
