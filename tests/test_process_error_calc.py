"""Test Module for ErrorCalc Process."""
from PIL import Image
import pytest
import torch
import torch.backends.cudnn as cudnn

from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.pipeline.error_calc import ErrorCalc


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
def network(device):
    """Pytest fixture of Network."""
    net = FCN8s(nclass=5, device=device)
    net.to(device)
    return net


@pytest.fixture()
def input_dict():
    """Pytest Fixture of dict with image bbox and masks with size 120x120."""
    result = {
        'image': Image.new("RGB", (120, 120), (255, 255, 255)),
        'bbox': torch.randint(0, 2, (1, 120, 120)),
        'masks': torch.randint(0, 2, (7, 120, 120))
    }
    for idx in range(result['masks'].shape[0]):
        result['masks'][idx, :, :] = result['masks'][idx, :, :] * torch.randint(0, 5, (1,))
    return result


class TestErrorCalc:
    """Test the methods of the ErrorCalcProcess."""

    def test_error_calc_set(self, network, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = ErrorCalc(network)

        # Run tested method
        process.set_inputs(**input_dict)

        # Asserts
        assert isinstance(process.in_image, Image.Image)
        assert isinstance(process.in_bbox, torch.Tensor)
        assert isinstance(process.in_masks, torch.Tensor)
        assert process.in_image.size == (120, 120)
        assert process.in_bbox.shape == (1, 120, 120)
        assert process.in_masks.shape == (7, 120, 120)

    def test_error_calc_get(self, network, input_dict):
        """Test the get_outputs Method of Process."""
        # Prepare
        process = ErrorCalc(network)
        process.set_inputs(**input_dict)

        # Run tested method
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'loss' in result.keys()
        assert result['loss'] is None

    def test_error_calc_update(self, network, input_dict):
        """Test the update method of Process."""
        # Prepare
        process = ErrorCalc(network)
        process.set_inputs(**input_dict)

        # Run tested method
        process.update()
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'loss' in result.keys()
        assert result['loss'].shape[0] == 7
