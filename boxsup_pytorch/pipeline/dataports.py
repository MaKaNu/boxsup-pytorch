"""Pipeline Port Module."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from .core import PipelineService


@dataclass
class BasePort:
    """Base Port for data transport."""

    data: Any = None
    sink_id: str = ""
    source_id: str = ""

    def transfer(self, service: PipelineService) -> None:
        if self.source_id:
            source = service.get_process(self.source_id)
            self.data = getattr(source, "out_" + self.data_name)
            self.initated = True
        if self.initated:
            sink = service.get_process(self.sink_id)
            setattr(sink, "in_" + self.data_name, self.data)
            sink.in_image = self.data
            return
        raise Exception("Port is not initiated!")

    def manual_insert(self, input_value: Any):
        self.initated = True
        self.data = input_value


class ImagePort(BasePort):
    """Image Port for transporting Images."""

    data_name: str = "image"
    data: Image = None

    def connect(self, service: PipelineService, sink_id: str, source_id: str = None):
        assert hasattr(service.get_process(sink_id), "in_image")
        if source_id:
            assert hasattr(service.get_process(sink_id), "out_image")
        self.sink_id = sink_id
        self.source_id = source_id


class MasksPort(BasePort):
    """Masks Port for transporting Masks."""

    data_name: str = "masks"
    data: npt.NDArray[np.float64] = None

    def connect(self, service: PipelineService, sink_id: str, source_id: str = None):
        assert hasattr(service.get_process(sink_id), "in_masks")
        if source_id:
            assert hasattr(service.get_process(sink_id), "out_masks")
        self.sink_id = sink_id
        self.source_id = source_id


class BBoxPort(BasePort):
    """Masks Port for transporting Masks."""

    data_name: str = "bbox"
    data: npt.NDArray[np.float64] = None

    def connect(self, service: PipelineService, sink_id: str, source_id: str = None):
        assert hasattr(service.get_process(sink_id), "in_bbox")
        if source_id:
            assert hasattr(service.get_process(sink_id), "out_bbox")
        self.sink_id = sink_id
        self.source_id = source_id
