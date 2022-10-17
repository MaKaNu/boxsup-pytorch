"""Test Module for ErrorCalc Process."""
from pathlib import Path

import pytest
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from boxsup_pytorch.data.dataset import BoxSupDataset
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
    net = FCN8s(nclass=7, device=device)
    net.to(device)
    return net


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
        'bbox': dataset[0][1][0],
        'masks': dataset[0][2][0:16]  # The First 16 Masks for testing
    }
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
        assert isinstance(process.in_image, torch.Tensor)
        assert isinstance(process.in_bbox, tuple)
        assert isinstance(process.in_bbox[0], int)
        assert isinstance(process.in_bbox[1], torch.Tensor)
        assert isinstance(process.in_masks, list)
        assert isinstance(process.in_masks[0], tuple)
        assert isinstance(process.in_masks[0][0], int)
        assert isinstance(process.in_masks[0][1], torch.Tensor)
        assert process.in_image.shape == (3, 224, 224)
        assert len(process.in_bbox) == 2
        assert len(process.in_masks) == 16

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
        assert len(result['loss']) == 16
