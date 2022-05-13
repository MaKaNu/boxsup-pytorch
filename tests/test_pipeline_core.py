"""Test Pipeline Core."""

import numpy as np
from PIL import Image
import pytest

from boxsup_pytorch.pipeline.core import PipelineService
from boxsup_pytorch.pipeline.dataports import BBoxPort, ImagePort, MasksPort
from boxsup_pytorch.pipeline.error_calc import ErrorCalc


@pytest.fixture()
def image():
    """Pytest fixture of 2x2 BoundingBox."""
    return Image.new("RGB", (120, 120), (255, 255, 255))


@pytest.fixture()
def bbox():
    """Pytest fixture of 2x2 BoundingBox."""
    return np.random.randint(0, 1, (1, 25, 25))


@pytest.fixture()
def masks():
    """Pytest fixture of 2x2 BoundingBox."""
    return np.random.randint(0, 5, (7, 25, 25))


@pytest.fixture()
def service():
    """Pytest fixture of PipelineServiceObject."""
    return PipelineService()


@pytest.fixture()
def process():
    """Pytest fixture of ProcessObject."""
    return ErrorCalc()


@pytest.fixture()
def port():
    """Pytest fixture of PortObject."""
    return ImagePort()


@pytest.fixture()
def ports():
    """Pytest fixture of PortObject."""
    return [ImagePort(), BBoxPort(), MasksPort()]


class TestPipelineService:
    """Testclass for PipelineService."""

    def test_registering(self, service, process, port):
        """Test the registering of PipelineService."""
        # Register
        error_calc_id = service.register_process(process)
        image_port_id = service.register_port(port, error_calc_id)
        assert image_port_id in service.dataports.keys()
        assert error_calc_id in service.processes.keys()

    def test_unregistering(self, service, process, port):
        """Test the registering of PipelineService."""
        # Register
        error_calc_id = service.register_process(process)
        image_port_id = service.register_port(port, error_calc_id)
        # Try to unregister blocked process
        with pytest.raises(Exception) as exc_info:
            service.unregister_process(error_calc_id)
        assert str(exc_info.value) == f"Process is blocked by ['{image_port_id}']"
        service.unregister_port(image_port_id)
        service.unregister_process(error_calc_id)
        assert error_calc_id not in service.processes.keys()
        assert image_port_id not in service.dataports.keys()

    def test_sending_data(self, service, process, ports, image, masks, bbox):
        """Test sending data of PipelineService."""
        # Register
        error_calc_id = service.register_process(process)
        image_port_id = service.register_port(ports[0], error_calc_id)
        box_port_id = service.register_port(ports[1], error_calc_id)
        masks_port_id = service.register_port(ports[2], error_calc_id)

        # Initiate Data Ports
        service.get_port(image_port_id).manual_insert(image)
        service.get_port(box_port_id).manual_insert(bbox)
        service.get_port(masks_port_id).manual_insert(masks)

        send_data_and_execute = [
            service.get_port(image_port_id).transfer,
            service.get_port(box_port_id).transfer,
            service.get_port(masks_port_id).transfer,
            service.get_process(error_calc_id).update,
        ]

        service.run_program(send_data_and_execute)

        result = service.get_process(error_calc_id).out_loss.shape[0]
        assert result == masks.shape[0]
