"""Tests for visual optical-flow metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologist.metrics import MetricEngine, VisualFlowMetric
from pathologist.metrics.surprise import ECODSurpriseScorer
from pathologist.schemas import BundleFormat, CameraFrameRef, MetricProfile, TrajectoryBundle


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_visual_flow_metric_computes_nonzero_flow_for_shifted_frames(tmp_path: Path) -> None:
    frames = _write_shifted_square_frames(tmp_path, camera_name="head")
    bundle = TrajectoryBundle(
        bundle_id="visual-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )

    result = VisualFlowMetric().compute(_context(bundle))

    assert result.name == "visual_flow"
    assert result.metadata["available"] is True
    assert result.metadata["configured_camera_names"] == ["head"]
    assert result.metadata["matched_camera_names"] == ["head"]
    assert result.metadata["frame_stride"] == 2
    assert result.metadata["resize"] == [128, 128]
    assert result.metadata["sample_camera_names"] == ["head", "head"]
    assert [sample.timestamp_sec for sample in result.samples] == [0.2, 0.4]
    assert all(sample.value > 0 for sample in result.samples)
    assert result.samples[0].window is not None
    assert result.samples[0].window.start_sec == 0.0
    assert result.samples[0].window.end_sec == 0.2


def test_visual_flow_metric_is_unavailable_when_configured_camera_is_missing(tmp_path: Path) -> None:
    frames = _write_shifted_square_frames(tmp_path, camera_name="wrist")
    bundle = TrajectoryBundle(
        bundle_id="visual-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )

    result = VisualFlowMetric().compute(
        _context(bundle, config={"metrics": {"visual_flow": {"camera_names": ["head"]}}})
    )

    assert result.name == "visual_flow"
    assert result.samples == []
    assert result.metadata["available"] is False
    assert result.metadata["configured_camera_names"] == ["head"]
    assert result.metadata["matched_camera_names"] == []
    assert "no frames found" in result.metadata["unavailable_reason"]


def test_visual_flow_is_in_default_engine_and_ecod_aggregate(tmp_path: Path) -> None:
    frames = _write_shifted_square_frames(tmp_path, camera_name="head")
    bundle = TrajectoryBundle(
        bundle_id="visual-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )

    metric_names = {metric.name for metric in MetricEngine().metrics}
    assert "visual_flow" in metric_names

    visual_flow = VisualFlowMetric().compute(_context(bundle))
    profile = MetricProfile(
        bundle_id="visual-demo",
        metrics=[
            _source_metric("action_discontinuity"),
            _source_metric("motion_smoothness"),
            _source_metric("velocity_spike"),
            _source_metric("stalling"),
            visual_flow,
        ],
    )

    scored_profile = ECODSurpriseScorer(config={"surprise": {"enabled": True}}).append_to_profile(profile)
    scored_names = {metric.name for metric in scored_profile.metrics}
    aggregate = next(metric for metric in scored_profile.metrics if metric.name == "ecod_aggregate_score")

    assert "visual_flow_ecod_score" in scored_names
    assert "visual_flow" in aggregate.metadata["source_metric_names"]


def _write_shifted_square_frames(tmp_path: Path, camera_name: str) -> list[CameraFrameRef]:
    camera_dir = tmp_path / camera_name
    camera_dir.mkdir()
    frames = []
    for index, offset in enumerate((8, 16, 24, 32, 40)):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[24:44, offset:offset + 20] = 255
        image_path = camera_dir / f"{index:06d}.jpg"
        assert cv2.imwrite(str(image_path), image)
        frames.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=index * 0.1,
                image_path=image_path,
                frame_index=index,
            )
        )
    return frames


def _source_metric(name: str):
    from pathologist.schemas import MetricResult, MetricSample

    return MetricResult(
        name=name,
        description=f"{name} source metric",
        unit="source_unit",
        samples=[
            MetricSample(timestamp_sec=0.0, value=1.0),
            MetricSample(timestamp_sec=0.1, value=2.0),
            MetricSample(timestamp_sec=0.2, value=3.0),
        ],
    )


def _context(bundle: TrajectoryBundle, config: dict | None = None):
    from pathologist.metrics.base import MetricComputationContext

    return MetricComputationContext(bundle=bundle, config=config or {})
