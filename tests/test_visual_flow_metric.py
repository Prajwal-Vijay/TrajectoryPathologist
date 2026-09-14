"""Tests for visual optical-flow metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologist.metrics import MetricEngine, VisualFlowMetric
from pathologist.metrics.robot_masking import (
    GroundedSAM2RobotArmMasker,
    RobotMaskBatch,
    _deduplicate_detections,
)
from pathologist.metrics.surprise import ECODSurpriseScorer
from pathologist.metrics.vision import _aggregate_flow_magnitudes
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


def test_visual_flow_can_exclude_grounded_sam2_robot_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_shifted_square_frames(tmp_path, camera_name="head")
    bundle = TrajectoryBundle(
        bundle_id="masked-visual-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )

    class FakeRobotMasker:
        def __init__(self, config):
            self.config = config

        def masks_for_paths(self, paths):
            masks = []
            for _ in paths:
                mask = np.zeros((80, 80), dtype=bool)
                mask[:, :64] = True
                masks.append(mask)
            return RobotMaskBatch(
                masks=masks,
                metadata={"method": "fake_grounded_sam2"},
            )

    monkeypatch.setattr(
        "pathologist.metrics.vision.GroundedSAM2RobotArmMasker",
        FakeRobotMasker,
    )
    unmasked = VisualFlowMetric().compute(
        _context(
            bundle,
            config={"metrics": {"visual_flow": {"frame_stride": 1}}},
        )
    )
    masked = VisualFlowMetric().compute(
        _context(
            bundle,
            config={
                "metrics": {
                    "visual_flow": {
                        "frame_stride": 1,
                        "robot_masking": {
                            "enabled": True,
                            "prompt": "robot arm. robot gripper.",
                            "dilation_px": 2,
                        },
                    }
                }
            },
        )
    )

    assert masked.metadata["robot_masking_enabled"] is True
    assert masked.metadata["robot_mask_metadata"]["head"]["method"] == "fake_grounded_sam2"
    assert masked.metadata["robot_mask_dilation_px"] == 2
    assert masked.metadata["mean_robot_mask_fraction"] > 0.8
    assert max(sample.value for sample in masked.samples) < max(
        sample.value for sample in unmasked.samples
    )


def test_top_fraction_flow_mean_focuses_on_largest_magnitudes() -> None:
    magnitudes = np.arange(100, dtype=np.float32)

    assert _aggregate_flow_magnitudes(
        magnitudes,
        aggregation="mean",
        top_fraction=0.02,
    ) == pytest.approx(49.5)
    assert _aggregate_flow_magnitudes(
        magnitudes,
        aggregation="top_fraction_mean",
        top_fraction=0.02,
    ) == pytest.approx(98.5)


def test_visual_flow_takes_timestamp_aligned_max_across_cameras(
    tmp_path: Path,
) -> None:
    head_frames = _write_shifted_square_frames(tmp_path, camera_name="head")
    wrist_frames = _write_shifted_square_frames(tmp_path, camera_name="wrist")
    bundle = TrajectoryBundle(
        bundle_id="multi-camera-visual-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=[*head_frames, *wrist_frames],
    )
    shared_config = {
        "frame_stride": 1,
        "resize": [128, 128],
        "spatial_aggregation": "top_fraction_mean",
        "top_fraction": 0.02,
    }

    head = VisualFlowMetric().compute(
        _context(
            bundle,
            config={
                "metrics": {
                    "visual_flow": {**shared_config, "camera_names": ["head"]}
                }
            },
        )
    )
    wrist = VisualFlowMetric().compute(
        _context(
            bundle,
            config={
                "metrics": {
                    "visual_flow": {**shared_config, "camera_names": ["wrist"]}
                }
            },
        )
    )
    combined = VisualFlowMetric().compute(
        _context(
            bundle,
            config={
                "metrics": {
                    "visual_flow": {
                        **shared_config,
                        "camera_names": ["head", "wrist"],
                    }
                }
            },
        )
    )

    expected = [
        max(float(head_sample.value), float(wrist_sample.value))
        for head_sample, wrist_sample in zip(head.samples, wrist.samples)
    ]
    assert [float(sample.value) for sample in combined.samples] == pytest.approx(
        expected
    )
    assert combined.metadata["camera_aggregation"] == (
        "max_after_linear_time_alignment"
    )
    assert combined.metadata["camera_reference_timeline"] == "head"
    assert combined.metadata["camera_flow_sample_counts"] == {
        "head": 4,
        "wrist": 4,
    }


def test_robot_masking_redetects_at_each_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [frame.image_path for frame in _write_shifted_square_frames(tmp_path, "head")]
    masker = GroundedSAM2RobotArmMasker({"chunk_size_frames": 2})
    detection_calls = []
    tracking_chunk_sizes = []

    monkeypatch.setattr(masker, "_load_models", lambda: None)

    def fake_detect(image):
        detection_calls.append(image.size)
        return [[0.0, 0.0, 20.0, 20.0]]

    def fake_track(chunk_paths, boxes):
        tracking_chunk_sizes.append(len(chunk_paths))
        return [np.zeros((80, 80), dtype=bool) for _ in chunk_paths]

    monkeypatch.setattr(masker, "_detect_robot_boxes", fake_detect)
    monkeypatch.setattr(masker, "_track_boxes", fake_track)

    result = masker.masks_for_paths(paths)

    assert len(detection_calls) == 3
    assert tracking_chunk_sizes == [2, 2, 1]
    assert result.metadata["chunk_detection_counts"] == [1, 1, 1]
    assert result.metadata["chunk_count"] == 3


def test_robot_detection_suppression_has_no_numerical_cap() -> None:
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [0.1, 0.1, 10.1, 10.1],
        [20.0, 0.0, 30.0, 10.0],
        [40.0, 0.0, 50.0, 10.0],
        [60.0, 0.0, 70.0, 10.0],
        [80.0, 0.0, 90.0, 10.0],
    ]

    kept = _deduplicate_detections(
        boxes,
        [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
        iou_threshold=0.8,
    )

    assert kept == [boxes[1], boxes[2], boxes[3], boxes[4], boxes[5]]


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
