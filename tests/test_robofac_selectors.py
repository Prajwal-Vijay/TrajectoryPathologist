"""Tests for RoboFAC selector behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pathologist.qa import robofac_selectors
from pathologist.qa.robofac_selectors import (
    _representative_frame_for_proposal,
    _top_peaks,
    select_kite_flow_keyframes,
    select_metric_keyframes,
)
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


def test_representative_frame_for_proposal_prefers_proposal_peak_sec() -> None:
    frames = [
        CameraFrameRef("rgb", 9.8, Path("/tmp/0098.jpg"), frame_index=98),
        CameraFrameRef("rgb", 10.0, Path("/tmp/0100.jpg"), frame_index=100),
        CameraFrameRef("rgb", 10.4, Path("/tmp/0104.jpg"), frame_index=104),
    ]
    proposal = SimpleNamespace(metadata={"peak_sec": 10.38}, metric_events=[])

    representative = _representative_frame_for_proposal(
        frames,
        TimeWindow(9.9, 10.3),
        proposal=proposal,
    )

    assert representative is not None
    assert representative.frame_index == 104


def test_representative_frame_for_proposal_uses_metric_event_peak_sec_fallback() -> None:
    frames = [
        CameraFrameRef("rgb", 9.8, Path("/tmp/0098.jpg"), frame_index=98),
        CameraFrameRef("rgb", 10.0, Path("/tmp/0100.jpg"), frame_index=100),
        CameraFrameRef("rgb", 10.4, Path("/tmp/0104.jpg"), frame_index=104),
    ]
    metric_event = SimpleNamespace(value={"peak_sec": 9.82})
    proposal = SimpleNamespace(metadata={}, metric_events=[metric_event])

    representative = _representative_frame_for_proposal(
        frames,
        TimeWindow(9.9, 10.3),
        proposal=proposal,
    )

    assert representative is not None
    assert representative.frame_index == 98


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

    resize_calls: list[tuple[int, int]] = []

    class FakeCV2:
        COLOR_BGR2GRAY = 0

        @staticmethod
        def imread(path: str) -> int:
            return int(Path(path).stem.split("_")[-1])

        @staticmethod
        def resize(image: int, size: tuple[int, int]) -> int:
            resize_calls.append(size)
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
    assert set(resize_calls) == {(128, 128)}

    padding_stats = {}
    selected_with_padding = select_kite_flow_keyframes(
        frames,
        frame_budget=5,
        config={"robofac_qa": {"kite_flow": {"frame_stride": 2, "nms_radius": 5}}},
        selection_stats=padding_stats,
        pad_to_budget=True,
    )

    assert [frame.frame_index for frame in selected_with_padding] == [0, 2, 9, 14, 19]
    assert padding_stats["method_frame_count"] == 2
    assert padding_stats["uniform_padded_frame_count"] == 3
    assert padding_stats["unfilled_frame_count"] == 0

    selected_without_padding = select_kite_flow_keyframes(
        frames,
        frame_budget=5,
        config={"robofac_qa": {"kite_flow": {"frame_stride": 2, "nms_radius": 5}}},
    )
    assert [frame.frame_index for frame in selected_without_padding] == [2, 14]


def test_kite_motion_nms_keeps_zero_motion_candidates_like_official_code() -> None:
    assert _top_peaks([0.0] * 7, count=2, suppression_radius=5) == [0, 6]


def test_metric_padding_divides_trajectory_by_remaining_budget(monkeypatch) -> None:
    frames = [
        CameraFrameRef("rgb", index / 10.0, Path(f"/tmp/frame_{index}.jpg"), frame_index=index)
        for index in range(20)
    ]
    proposals = [
        SimpleNamespace(
            candidate_frames=[frames[index]],
            time_window=TimeWindow(frames[index].timestamp_sec, frames[index].timestamp_sec),
            metadata={"peak_sec": frames[index].timestamp_sec},
            metric_events=[],
        )
        for index in (5, 10, 15)
    ]

    class FakeMetricEventSelector:
        def __init__(self, **_kwargs) -> None:
            pass

        def select(self, *_args, **_kwargs):
            return proposals

    monkeypatch.setattr(robofac_selectors, "MetricEventSelector", FakeMetricEventSelector)
    stats = {}

    selected = select_metric_keyframes(
        SimpleNamespace(camera_frames=frames),
        metric_profile=object(),
        frame_budget=6,
        config={},
        selection_stats=stats,
    )

    assert [frame.frame_index for frame in selected] == [0, 5, 9, 10, 15, 19]
    assert stats["metric_frame_count"] == 3
    assert stats["uniform_padded_frame_count"] == 3
    assert stats["unfilled_frame_count"] == 0
