"""Input trajectory bundle schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathologist.schemas.common import Modality, SchemaBase, StrEnum, TimeWindow, require_non_empty
from pathologist.schemas.task_contract import TaskContract


class BundleFormat(StrEnum):
    """Known external storage formats."""

    ROBOEVAL_VLM_BUNDLE = "roboeval_vlm_bundle"
    VIFAILBACK_HDF5_VIDEO = "vifailback_hdf5_video"
    ROBOFAC_HDF5_VIDEO = "robofac_hdf5_video"
    RLDS = "rlds"
    DROID = "droid"
    ROBOTWIN = "robotwin"
    CUSTOM = "custom"


@dataclass(slots=True)
class ModalityStream:
    """One timestamped stream within a trajectory bundle."""

    name: str
    modality: Modality
    timestamps_sec: list[float]
    data_path: Path | None = None
    records: list[dict[str, Any]] | None = None
    sample_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.name, "ModalityStream.name")
        if self.records is None and self.data_path is None:
            raise ValueError("ModalityStream requires either records or data_path.")
        if self.timestamps_sec and any(t < 0 for t in self.timestamps_sec):
            raise ValueError("timestamps_sec must be non-negative.")


@dataclass(slots=True)
class CameraFrameRef:
    """Reference to a saved image frame."""

    camera_name: str
    timestamp_sec: float
    image_path: Path
    frame_index: int | None = None
    time_window: TimeWindow | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.camera_name, "CameraFrameRef.camera_name")
        if self.timestamp_sec < 0:
            raise ValueError("CameraFrameRef.timestamp_sec must be non-negative.")


@dataclass(slots=True)
class TrajectoryBundle(SchemaBase):
    """Dataset-agnostic representation of one robot trajectory."""

    bundle_id: str = ""
    source_format: str = BundleFormat.CUSTOM
    root_path: Path | None = None
    task_contract: TaskContract | None = None
    duration_sec: float | None = None
    streams: list[ModalityStream] = field(default_factory=list)
    camera_frames: list[CameraFrameRef] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.bundle_id, "TrajectoryBundle.bundle_id")
        if self.duration_sec is not None and self.duration_sec < 0:
            raise ValueError("TrajectoryBundle.duration_sec must be non-negative.")
