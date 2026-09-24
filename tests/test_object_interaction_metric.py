"""Tests for Grounded-SAM2 and CoTracker object interaction motion."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pathologist.metrics import MetricEngine, ObjectInteractionMotionMetric
from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.object_interaction import (
    GroundedSAM2InteractionSegmenter,
    InteractionMotionRecord,
    _aggregate_camera_records,
    _select_nearest_object_box,
    camera_compensated_motion_records,
)
from pathologist.metrics.robot_masking import RobotMaskBatch
from pathologist.metrics.surprise import DEFAULT_SURPRISE_SOURCE_METRICS
from pathologist.schemas import BundleFormat, CameraFrameRef, TrajectoryBundle


np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _background_grid() -> np.ndarray:
    return np.asarray(
        [
            [x, y]
            for y in (10.0, 35.0, 65.0, 90.0)
            for x in (10.0, 50.0, 90.0)
        ]
    )


def test_background_homography_removes_camera_translation() -> None:
    background = _background_grid()
    object_points = np.asarray([[60.0, 45.0], [60.0, 55.0]])
    gripper_points = np.asarray([[40.0, 45.0], [40.0, 55.0]])
    first = np.concatenate([object_points, gripper_points, background], axis=0)
    camera_translation = np.asarray([5.0, 3.0])
    second = first + camera_translation
    second[:2, 0] += 2.0
    tracks = np.stack([first, second], axis=0)
    visibility = np.ones(tracks.shape[:2], dtype=bool)

    records = camera_compensated_motion_records(
        tracks,
        visibility,
        object_indices=np.arange(0, 2),
        gripper_indices=np.arange(2, 4),
        background_indices=np.arange(4, 4 + len(background)),
        timestamps_sec=[0.0, 0.5],
        frame_width=100,
        frame_height=100,
        ransac_threshold_px=0.5,
    )

    assert len(records) == 1
    expected = 2.0 / 0.5 / np.hypot(100.0, 100.0)
    assert records[0].components["object_residual_speed"] == pytest.approx(expected)
    assert records[0].components["gripper_object_relative_speed"] == pytest.approx(
        expected
    )
    assert records[0].components["gripper_object_separation_speed"] == pytest.approx(
        expected
    )
    assert records[0].homography_inlier_fraction == pytest.approx(1.0)


def test_object_residual_speed_keeps_rotational_point_motion() -> None:
    background = _background_grid()
    first = np.concatenate(
        [
            np.asarray([[40.0, 50.0], [60.0, 50.0]]),
            np.asarray([[20.0, 45.0], [20.0, 55.0]]),
            background,
        ],
        axis=0,
    )
    second = first.copy()
    second[:2] = np.asarray([[50.0, 40.0], [50.0, 60.0]])

    records = camera_compensated_motion_records(
        np.stack([first, second], axis=0),
        np.ones((2, 4 + len(background)), dtype=bool),
        object_indices=np.arange(0, 2),
        gripper_indices=np.arange(2, 4),
        background_indices=np.arange(4, 4 + len(background)),
        timestamps_sec=[0.0, 1.0],
        frame_width=100,
        frame_height=100,
        ransac_threshold_px=0.5,
    )

    expected = np.hypot(10.0, 10.0) / np.hypot(100.0, 100.0)
    assert records[0].components["object_residual_speed"] == pytest.approx(expected)
    assert records[0].components["gripper_object_relative_speed"] == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_degenerate_background_tracks_skip_pair() -> None:
    background = np.asarray([[float(index), 10.0] for index in range(12)])
    first = np.concatenate(
        [
            np.asarray([[40.0, 50.0], [42.0, 50.0]]),
            np.asarray([[20.0, 50.0], [22.0, 50.0]]),
            background,
        ],
        axis=0,
    )

    records = camera_compensated_motion_records(
        np.stack([first, first + np.asarray([2.0, 1.0])], axis=0),
        np.ones((2, len(first)), dtype=bool),
        object_indices=np.arange(0, 2),
        gripper_indices=np.arange(2, 4),
        background_indices=np.arange(4, len(first)),
        timestamps_sec=[0.0, 0.5],
        frame_width=100,
        frame_height=100,
        ransac_threshold_px=0.5,
    )

    assert records == []


def test_camera_aggregation_keeps_components_from_scalar_winner() -> None:
    first = InteractionMotionRecord(
        timestamp_sec=1.0,
        start_sec=0.5,
        components={
            "object_residual_speed": 10.0,
            "gripper_object_relative_speed": 0.0,
            "gripper_object_separation_speed": 0.0,
        },
        homography_inlier_fraction=1.0,
        visible_object_point_count=8,
        visible_gripper_point_count=8,
        visible_background_point_count=16,
    )
    second = InteractionMotionRecord(
        timestamp_sec=1.0,
        start_sec=0.5,
        components={
            "object_residual_speed": 0.0,
            "gripper_object_relative_speed": 9.0,
            "gripper_object_separation_speed": 9.0,
        },
        homography_inlier_fraction=1.0,
        visible_object_point_count=8,
        visible_gripper_point_count=8,
        visible_background_point_count=16,
    )

    samples, cameras, components, _ = _aggregate_camera_records(
        {"head": [first], "wrist": [second]},
        max_alignment_gap_sec=0.5,
    )

    assert [sample.value for sample in samples] == [10.0]
    assert cameras == ["head"]
    assert components == {
        "object_residual_speed": [10.0],
        "gripper_object_relative_speed": [0.0],
        "gripper_object_separation_speed": [0.0],
    }


def test_object_initializer_prefers_nearby_non_gripper_box() -> None:
    gripper = [40.0, 40.0, 60.0, 60.0]
    overlapping_gripper = [41.0, 41.0, 59.0, 59.0]
    nearby_object = [62.0, 42.0, 75.0, 58.0]
    distant_object = [5.0, 5.0, 20.0, 20.0]

    selected = _select_nearest_object_box(
        [overlapping_gripper, distant_object, nearby_object],
        gripper_box=gripper,
        image_width=100,
        image_height=100,
        max_area_fraction=0.35,
        max_gripper_iou=0.7,
    )

    assert selected == nearby_object


def test_object_initializer_rejects_ambiguous_nearby_objects() -> None:
    selected = _select_nearest_object_box(
        [
            [62.0, 42.0, 75.0, 58.0],
            [25.0, 42.0, 38.0, 58.0],
        ],
        gripper_box=[40.0, 40.0, 60.0, 60.0],
        image_width=100,
        image_height=100,
        max_area_fraction=0.35,
        max_gripper_iou=0.7,
        ambiguity_margin_px=1.0,
    )

    assert selected is None


def test_overlapping_chunks_reseed_the_previous_object_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"{index}.png" for index in range(3)]
    for path in paths:
        Image.new("RGB", (100, 100)).save(path)
    segmenter = GroundedSAM2InteractionSegmenter({"chunk_size_frames": 3})
    gripper_mask = np.zeros((100, 100), dtype=bool)
    gripper_mask[40:60, 40:60] = True

    def fake_gripper_masks(image_paths):
        return RobotMaskBatch(
            masks=[gripper_mask.copy() for _ in image_paths],
            metadata={"method": "fake"},
        )

    detection_calls = []

    def fake_detect(image):
        detection_calls.append(image.size)
        return [[62.0, 42.0, 75.0, 58.0]]

    def fake_track(chunk_paths, boxes):
        box = boxes[0]
        output = []
        for _ in chunk_paths:
            mask = np.zeros((100, 100), dtype=bool)
            mask[
                int(box[1]):int(box[3]),
                int(box[0]):int(box[2]),
            ] = True
            output.append(mask)
        return output

    monkeypatch.setattr(
        segmenter.gripper_masker,
        "masks_for_paths",
        fake_gripper_masks,
    )
    monkeypatch.setattr(segmenter.object_masker, "_read_cached_masks", lambda paths: None)
    monkeypatch.setattr(segmenter.object_masker, "_write_cached_masks", lambda *args: None)
    monkeypatch.setattr(segmenter.object_masker, "_load_models", lambda: None)
    monkeypatch.setattr(segmenter.object_masker, "_detect_robot_boxes", fake_detect)
    monkeypatch.setattr(segmenter.object_masker, "_track_boxes", fake_track)

    segmenter.masks_for_paths(paths[:2])
    second = segmenter.masks_for_paths(paths[1:])

    assert len(detection_calls) == 1
    assert second.metadata["continuity_seeded_chunk_count"] == 1
    assert second.object_masks[0].any()


def test_metric_emits_independent_ecod_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        CameraFrameRef(
            camera_name="head",
            timestamp_sec=index * 0.5,
            image_path=Path(f"/unused/{index}.jpg"),
            frame_index=index,
        )
        for index in range(4)
    ]
    bundle = TrajectoryBundle(
        bundle_id="interaction-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )

    class FakeTracker:
        def __init__(self, config):
            self.config = config
            self.device = "cpu"

        def load(self):
            return None

    monkeypatch.setattr(
        "pathologist.metrics.object_interaction.CoTrackerRunner",
        FakeTracker,
    )
    records = [
        InteractionMotionRecord(
            timestamp_sec=index * 0.5,
            start_sec=(index - 1) * 0.5,
            components={
                "object_residual_speed": float(index),
                "gripper_object_relative_speed": float(index + 1),
                "gripper_object_separation_speed": float(index + 2),
            },
            homography_inlier_fraction=1.0,
            visible_object_point_count=8,
            visible_gripper_point_count=4,
            visible_background_point_count=16,
        )
        for index in range(1, 4)
    ]

    def fake_compute(*args, **kwargs):
        del args, kwargs
        return records, [{"method": "fake"}]

    monkeypatch.setattr(
        "pathologist.metrics.object_interaction._compute_camera_records",
        fake_compute,
    )
    result = ObjectInteractionMotionMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={
                "metrics": {
                    "object_interaction_motion": {
                        "enabled": True,
                        "camera_names": ["head"],
                    }
                }
            },
        )
    )

    assert result.metadata["available"] is True
    assert result.metadata["ecod_component_values"] == {
        "object_residual_speed": [1.0, 2.0, 3.0],
        "gripper_object_relative_speed": [2.0, 3.0, 4.0],
        "gripper_object_separation_speed": [3.0, 4.0, 5.0],
    }
    assert [sample.value for sample in result.samples] == [3.0, 4.0, 5.0]
    assert result.metadata["model_training"] == (
        "pretrained_frozen_no_task_finetuning"
    )


def test_metric_is_registered_but_disabled_by_default() -> None:
    metric_names = {metric.name for metric in MetricEngine().metrics}

    assert "object_interaction_motion" in metric_names
    assert "object_interaction_motion" in DEFAULT_SURPRISE_SOURCE_METRICS
