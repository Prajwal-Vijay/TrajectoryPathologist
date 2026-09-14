"""RT-1 RoboFine-Bench adapter: xyz + quaternion state at 3 Hz."""

from __future__ import annotations

from pathologist.adapters.robofine_lerobot import RoboFineDatasetSpec, RoboFineLeRobotAdapter


class RoboFineRT1Adapter(RoboFineLeRobotAdapter):
    """Unpack RT-1 `observation.state` as xyz + unit quaternion + gripper."""

    spec = RoboFineDatasetSpec(
        dataset="RT-1",
        fps=3.0,
        orientation="quaternion",
        default_view="image",
        state_names=("x", "y", "z", "rx", "ry", "rz", "rw", "gripper"),
    )
