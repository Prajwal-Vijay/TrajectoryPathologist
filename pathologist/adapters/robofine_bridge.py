"""BridgeDataV2 RoboFine-Bench adapter: xyz + Euler state at 5 Hz."""

from __future__ import annotations

from pathologist.adapters.robofine_lerobot import RoboFineDatasetSpec, RoboFineLeRobotAdapter


class RoboFineBridgeAdapter(RoboFineLeRobotAdapter):
    """Unpack Bridge `observation.state` as xyz + rpy + pad + gripper."""

    spec = RoboFineDatasetSpec(
        dataset="BridgeDataV2",
        fps=5.0,
        orientation="euler",
        default_view="image_0",
        state_names=("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
    )
