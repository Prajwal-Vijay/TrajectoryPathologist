"""Tests for RoboFAC selector behavior."""

from __future__ import annotations

from pathlib import Path

from pathologist.qa import robofac_selectors
from pathologist.qa.robofac_selectors import _representative_frame_for_proposal, select_kite_flow_keyframes
from pathologist.schemas import CameraFrameRef, TimeWindow


def test_representative_frame_for_proposal_picks_closest_to_center() -> None:
    frames = [
        CameraFrameRef("rgb", 9.8, Path("/tmp/0098.jpg"), frame_index=98),
        CameraFrameRef("rgb", 10.0, Path("/tmp/0100.jpg"), frame_index=100),
        CameraFrameRef("rgb", 10.4, Path("/tmp/0104.jpg"), frame_index=104),
    ]

    representative = _representative_frame_for_proposal(frames, TimeWindow(9.9, 10.3))

    assert representative is not None
    assert representative.frame_index == 100


def test_kite_flow_selector_uses_stride_and_fixed_sample_nms(monkeypatch) -> None:
    frames = [
        CameraFrameRef("rgb", index / 10.0, Path(f"/tmp/frame_{index}.jpg"), frame_index=index)
        for index in range(20)
    ]

    class FakeFlow:
        def __init__(self, score: float) -> None:
            self.score = score

        def __getitem__(self, _key: object) -> float:
            return self.score

    class FakeMagnitude:
        def __init__(self, score: float) -> None:
            self.score = score

        def mean(self) -> float:
            return self.score

    class FakeCV2:
        COLOR_BGR2GRAY = 0

        @staticmethod
        def imread(path: str) -> int:
            return int(Path(path).stem.split("_")[-1])

        @staticmethod
        def resize(image: int, _size: tuple[int, int]) -> int:
            return image

        @staticmethod
        def cvtColor(image: int, _mode: int) -> int:
            return image

        @staticmethod
        def calcOpticalFlowFarneback(previous: int, current: int, *_args: object) -> FakeFlow:
            return FakeFlow(abs(current - previous))

        @staticmethod
        def cartToPolar(x_value: float, _y_value: float) -> tuple[FakeMagnitude, None]:
            return FakeMagnitude(x_value), None

    monkeypatch.setattr(robofac_selectors, "_import_cv2", lambda: FakeCV2)

    selected = select_kite_flow_keyframes(
        frames,
        frame_budget=2,
        config={"robofac_qa": {"kite_flow": {"frame_stride": 2, "nms_radius": 5}}},
    )

    assert [frame.frame_index for frame in selected] == [2, 14]
