"""Tests for frozen DINOv2 patch-level temporal novelty."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologist.metrics import FrozenDinoPatchNoveltyMetric, MetricEngine
from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.surprise import DEFAULT_SURPRISE_SOURCE_METRICS
from pathologist.metrics.visual_novelty import (
    symmetric_local_patch_novelty,
    top_fraction_mean,
)
from pathologist.schemas import BundleFormat, CameraFrameRef, TrajectoryBundle


torch = pytest.importorskip("torch")


def test_local_correspondence_absorbs_a_one_patch_swap() -> None:
    previous = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    current = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    same_position = symmetric_local_patch_novelty(previous, current, radius=0)
    locally_matched = symmetric_local_patch_novelty(previous, current, radius=1)

    assert same_position.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert locally_matched.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_symmetric_novelty_detects_changed_patch_content() -> None:
    previous = torch.tensor([[[1.0, 0.0]]])
    current = torch.tensor([[[0.0, 1.0]]])

    novelty = symmetric_local_patch_novelty(previous, current, radius=0)

    assert novelty.tolist() == pytest.approx([1.0, 1.0])


def test_top_fraction_mean_keeps_only_largest_patch_changes() -> None:
    assert top_fraction_mean(torch.tensor([0.0, 1.0, 2.0, 3.0]), 0.5) == pytest.approx(
        2.5
    )


def test_metric_builds_temporal_samples_from_frozen_patch_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        CameraFrameRef(
            camera_name="head",
            timestamp_sec=index * 0.5,
            image_path=Path(f"/unused/{index}.jpg"),
            frame_index=index,
        )
        for index in range(3)
    ]
    bundle = TrajectoryBundle(
        bundle_id="dino-demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=frames,
    )
    first = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    grids = [first, first.clone(), -first]

    monkeypatch.setattr(
        "pathologist.metrics.visual_novelty._load_dinov2_resources",
        lambda model_id, requested_device: {"device": "cpu"},
    )

    def fake_embeddings(
        sampled_frames,
        *,
        resources,
        batch_size,
        image_read_failures,
    ):
        del resources, batch_size, image_read_failures
        for frame in sampled_frames:
            yield frame, grids[int(frame.frame_index)]

    monkeypatch.setattr(
        "pathologist.metrics.visual_novelty._iter_patch_embeddings",
        fake_embeddings,
    )
    result = FrozenDinoPatchNoveltyMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={
                "metrics": {
                    "visual_patch_novelty": {
                        "enabled": True,
                        "camera_names": ["head"],
                        "sample_period_sec": 0.5,
                        "local_match_radius": 0,
                        "top_fraction": 0.5,
                    }
                }
            },
        )
    )

    assert result.metadata["available"] is True
    assert result.metadata["model_frozen"] is True
    assert result.metadata["patch_grid_shapes"] == {"head": [2, 2]}
    assert [sample.timestamp_sec for sample in result.samples] == [0.5, 1.0]
    assert result.samples[0].value == pytest.approx(0.0)
    assert result.samples[1].value == pytest.approx(2.0)


def test_metric_is_registered_as_a_default_ecod_source() -> None:
    metric_names = {metric.name for metric in MetricEngine().metrics}

    assert "visual_patch_novelty" in metric_names
    assert "visual_patch_novelty" in DEFAULT_SURPRISE_SOURCE_METRICS
