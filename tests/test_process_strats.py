"""Test Module for GreedyStrat Process."""
from PIL import Image
import pytest
import torch
import torch.backends.cudnn as cudnn

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
    net = FCN8s(nclass=5, device=device)
    net.to(device)
    process = ErrorCalc(net)
    return process


@pytest.fixture()
def input_dict():
    """Pytest Fixture of dict with image bbox and masks with size 120x120."""
    result = {
        'image': Image.new("RGB", (120, 120), (255, 255, 255)),
        'bboxes': torch.randint(0, 2, (5, 120, 120)),
        'masks': torch.randint(0, 2, (7, 120, 120))
    }
    for idx in range(result['masks'].shape[0]):
        result['masks'][idx, :, :] = result['masks'][idx, :, :] * torch.randint(0, 5, (1,))
    for idx in range(result['bboxes'].shape[0]):
        result['bboxes'][idx, :, :] = result['bboxes'][idx, :, :] * torch.randint(0, 5, (1,))

    return result


class TestGreedyStrat():
    """Test the methods of the GreedyStratProcess."""

    def test_greedy_strat_set(self, error_calc, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = GreedyStrat(error_calc)

        # Run tested method
        process.set_inputs(**input_dict)

        # Asserts
        assert isinstance(process.in_image, Image.Image)
        assert isinstance(process.in_bboxes, torch.Tensor)
        assert isinstance(process.in_masks, torch.Tensor)
        assert process.in_image.size == (120, 120)
        assert process.in_bboxes.shape == (5, 120, 120)
        assert process.in_masks.shape == (7, 120, 120)

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

    def test_greedy_strat__reduce_masks(self, error_calc, input_dict):
        """Test the update _reduce_masks method of Process."""
        # Prepare
        process = GreedyStrat(error_calc)
        masks = input_dict['masks']

        # Run tested method
        result = process._reduce_masks(masks)

        # Asserts
        assert result.dtype == torch.int64
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([120, 120])

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
        assert result['labelmask'].shape == torch.Size((120, 120))


class TestMiniMaxStrat():
    """Test the methods of the GreedyStratProcess."""

    def test_minimax_strat_set(self, error_calc, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = MiniMaxStrat(error_calc)

        # Run tested method
        process.set_inputs(**input_dict)

        # Asserts
        assert isinstance(process.in_image, Image.Image)
        assert isinstance(process.in_bboxes, torch.Tensor)
        assert isinstance(process.in_masks, torch.Tensor)
        assert process.in_image.size == (120, 120)
        assert process.in_bboxes.shape == (5, 120, 120)
        assert process.in_masks.shape == (7, 120, 120)

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

    def test_minimax_strat__reduce_masks(self, error_calc, input_dict):
        """Test the update _reduce_masks method of Process."""
        # Prepare
        process = MiniMaxStrat(error_calc)
        masks = input_dict['masks']

        # Run tested method
        result = process._reduce_masks(masks)

        # Asserts
        assert result.dtype == torch.int64
        assert result.device.type == "cpu"
        assert result.shape == torch.Size([120, 120])

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
        assert result['labelmask'].shape == torch.Size((120, 120))
