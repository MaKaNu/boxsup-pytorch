"""Test Module for ErrorCalc Process."""
import random

from PIL import Image, ImageDraw
import pytest
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms

from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.pipeline.error_calc import ErrorCalc
from boxsup_pytorch.utils.losses import Losses


torch.manual_seed(17)
random.seed(17)


def square_matrix(center, size=60, width=120, height=120):
    """Create a circle image based on center, width and height."""
    image = Image.new(mode="1", size=(width, height))
    draw = ImageDraw.Draw(image)
    x0 = center[0] - (size // 2)
    y0 = center[1] - (size // 2)
    x1 = center[0] + (size // 2) + 1
    y1 = center[1] + (size // 2) + 1
    draw.rectangle((x0, y0, x1, y1), fill=1)
    return transforms.ToTensor()(image).to(torch.long)


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
    net = FCN8s(nclass=3, device=device)
    net.to(device)
    return net


@pytest.fixture()
def network_mocked(mocker):
    """Pytest fixture of mocked Network."""
    mock_client_class = mocker.patch('boxsup_pytorch.pipeline.error_calc.FCN8s')
    net = mock_client_class.return_value
    return net


@pytest.fixture()
def losses():
    """Pytest fixture of Losses."""
    return Losses()


@pytest.fixture()
def input_dict():
    """Pytest Fixture of dict with image bbox and masks with size 120x120."""
    # Create bboxes
    center = (45, 60)
    bbox = square_matrix(center)

    # Create masks
    center1 = (75, 60)
    center2 = (75, 30)
    masks = [
        square_matrix(center1),
        square_matrix(center2) * 2
    ]
    result = {
        'image': Image.new("RGB", (120, 120), (255, 255, 255)),
        'bbox': bbox.squeeze(),
        'masks': torch.stack(masks).squeeze()
    }
    return result


def test_rectangle_image(capsys):
    """Test the ficture circle_matrix left to display it."""
    center = (45, 60)
    test_circle_matrix = square_matrix(center).to(torch.float32)
    scaled_matrix = F.interpolate(test_circle_matrix[None, :], size=(80*2//5, 80))
    height = scaled_matrix.shape[2]
    with capsys.disabled():
        print()
        for i in range(height):
            line = scaled_matrix[0, 0, i, :].to(int).tolist()
            line_str = ''.join(str(x) for x in line)
            print(line_str)
        print()


class TestErrorCalc:
    """Test the methods of the ErrorCalcProcess."""

    def test_error_calc_set(self, network_mocked, losses, input_dict):
        """Test the set_inputs Method of Process."""
        # Prepare
        process = ErrorCalc(network_mocked, losses)

        # Run tested method
        process.set_inputs(input_dict)

        # Asserts
        assert isinstance(process.in_image, Image.Image)
        assert isinstance(process.in_bbox, torch.Tensor)
        assert isinstance(process.in_masks, torch.Tensor)
        assert process.in_image.size == (120, 120)
        assert process.in_bbox.shape == (2, 120, 120)
        assert process.in_masks.shape == (7, 120, 120)

    def test_error_calc_get(self, network_mocked, losses, input_dict):
        """Test the get_outputs Method of Process, without calling the update."""
        # Prepare
        process = ErrorCalc(network_mocked, losses)
        process.set_inputs(input_dict)

        # Run tested method
        result = process.get_outputs()

        # Asserts
        assert isinstance(result, dict)
        assert 'loss' in result.keys()
        assert result['loss'] is None

    def test_error_calc_update(self, network, losses, input_dict):
        """Test the update method of Process and test the against the output with network."""
        # Prepare
        process = ErrorCalc(network, losses)
        process.set_inputs(input_dict)

        # Run tested method
        process.update()
        result = process.get_outputs()

        # targets
        target_0 = 1.7330515385
        target_1 = 0.8357251287

        # Asserts
        assert isinstance(result, dict)
        assert 'loss' in result.keys()
        assert result['loss'] is not None
        assert result['loss'][0] == pytest.approx(target_0)
        assert result['loss'][1] == pytest.approx(target_1)

    def test_error_calc_update_mocked(self, network_mocked, losses, input_dict, mocker):
        """Test the update method of Process and test the against the output with mocked network."""
        # Mocking
        network.device = mocker.MagicMock(return_value='cpu')

        # Prepare
        process = ErrorCalc(network_mocked, losses)
        process.set_inputs(input_dict)

        # Mocking Precess methods
        process._network_inference = mocker.MagicMock(return_value=torch.rand(1, 3, 120, 120))

        # Run tested method
        process.update()
        result = process.get_outputs()

        # targets
        target_0 = 1.2281949520
        target_1 = 1.2239211798

        # Asserts
        assert isinstance(result, dict)
        assert 'loss' in result.keys()
        assert result['loss'] is not None
        assert result['loss'][0] == pytest.approx(target_0)
        assert result['loss'][1] == pytest.approx(target_1)
