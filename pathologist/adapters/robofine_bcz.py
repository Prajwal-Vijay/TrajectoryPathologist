"""BC-Z RoboFine-Bench adapter: xyz + Euler state at 10 Hz."""

from __future__ import annotations

from pathologist.adapters.robofine_lerobot import RoboFineDatasetSpec, RoboFineLeRobotAdapter


class RoboFineBCZAdapter(RoboFineLeRobotAdapter):
    """Unpack BC-Z `observation.state` as xyz + rpy + pad + gripper, with π-wrap unwrapping."""

    spec = RoboFineDatasetSpec(
        dataset="BC-Z",
        fps=10.0,
        orientation="euler",
        default_view="image",
        state_names=("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
    )
