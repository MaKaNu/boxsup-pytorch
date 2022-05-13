"""Pipeline core module."""
from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class PipelineProcess(Protocol):
    """Process of a Pipeline."""

    def update(self):
        """Update the internal data."""
        ...  # pragma: no cover


@runtime_checkable
class PipelineDataport(Protocol):
    """Pipeline Connector Protocol."""

    sink_id: str = ""
    source_id: str = ""

    def connect(self, service: PipelineService, sink_id: str, source_id: str = None):
        """Connect between source and sink.

        Args:
            sink_id (str): ID of Sink Process
            source_id (str): ID of the Source Process
        """
        ...  # pragma: no cover

    def transfer(self, data=None):
        """Transfer Data from source to sink."""
        ...  # pragma: no cover


def generate_id(object: PipelineDataport | PipelineProcess):
    """Generate an ID of given length.

    Args:
        length (int, optional): length of ID. Defaults to 8.
        mode (str): either "port" | "proc"
    """
    if isinstance(object, PipelineProcess):
        mode = "proc"
    elif isinstance(object, PipelineDataport):
        mode = "port"
    else:
        raise Exception("object is not subclass of given protocol!")
    hashed_object = id(object)
    return f"{mode}-{hashed_object}"


class PipelineService:
    """The Core Pipeline service."""

    def __init__(self) -> None:
        """Pipeline Service Constructor."""
        self.dataports: Dict[str, PipelineDataport] = {}
        self.processes: Dict[str, PipelineProcess] = {}
        self.blocked: Dict[str, List[str]] = {}

    def register_process(self, process: PipelineProcess) -> str:
        process_id = generate_id(process)
        self.processes[process_id] = process
        return process_id

    def unregister_process(self, process_id: str) -> None:
        if self.blocked[process_id]:
            raise Warning(f"Process is blocked by {self.blocked[process_id]}")
        del self.processes[process_id]

    def get_process(self, process_id: str) -> PipelineProcess:
        return self.processes[process_id]

    def register_port(self, port: PipelineDataport, sink_id: str, source_id: str = None) -> str:
        port_id = generate_id(port)
        self.blocked.setdefault(sink_id, []).append(port_id)
        if source_id:
            self.blocked.setdefault(source_id, []).append(port_id)
        port.connect(self, sink_id, source_id)
        self.dataports[port_id] = port
        return port_id

    def unregister_port(self, port_id: str) -> None:
        port = self.get_port(port_id)
        sink_id = port.sink_id
        source_id = port.source_id
        self.blocked[sink_id].remove(port_id)
        if source_id:
            self.blocked[sink_id].remove(port_id)
        del self.dataports[port_id]

    def get_port(self, port_id) -> PipelineDataport:
        return self.dataports[port_id]

    def run_program(self, steps):
        print("=== Begin Program ===")
        for step in steps:
            if step.__name__ == "transfer":
                step(self)
            else:
                step()
        print("==== End Program ====")
