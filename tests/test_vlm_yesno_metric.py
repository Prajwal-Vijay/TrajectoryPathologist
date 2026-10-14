from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.engine import MetricEngine
from pathologist.metrics.vlm_yesno import CachedVLMYesNoMetric
from pathologist.schemas import BundleFormat, TrajectoryBundle


def _bundle() -> TrajectoryBundle:
    return TrajectoryBundle(
        bundle_id="episode_1",
        source_format=BundleFormat.CUSTOM,
        duration_sec=4.0,
        raw_metadata={"task_name": "insert_object"},
    )


def test_cached_vlm_metric_loads_centers_and_margins(tmp_path: Path) -> None:
    directory = tmp_path / "insert_object"
    directory.mkdir()
    np.savez_compressed(
        directory / "episode_1.npz",
        center_sec=np.asarray([0.9, 1.9, 2.9]),
        margin=np.asarray([-2.0, 3.0, -1.0]),
        yes_no_mass=np.asarray([0.9, 0.8, 0.4]),
    )
    context = MetricComputationContext(
        bundle=_bundle(),
        config={"vlm_yesno": {"cache_root": str(tmp_path)}},
    )

    metric = CachedVLMYesNoMetric().compute(context)

    assert metric.metadata["available"] is True
    assert [sample.timestamp_sec for sample in metric.samples] == [0.9, 1.9, 2.9]
    assert [sample.value for sample in metric.samples] == [-2.0, 3.0, -1.0]
    assert metric.metadata["yes_no_mass_below_0_5_count"] == 1


def test_cached_vlm_metric_marks_missing_cache_unavailable(tmp_path: Path) -> None:
    metric = CachedVLMYesNoMetric().compute(
        MetricComputationContext(
            bundle=_bundle(),
            config={"vlm_yesno": {"cache_root": str(tmp_path)}},
        )
    )

    assert metric.samples == []
    assert metric.metadata["available"] is False
    assert metric.metadata["unavailable_reason"] == "cache file is missing"


def test_metric_engine_registers_cached_vlm_only_when_configured(
    tmp_path: Path,
) -> None:
    without_vlm = MetricEngine(config={"surprise": {"enabled": False}}).compute(
        _bundle()
    )
    with_vlm = MetricEngine(
        config={
            "surprise": {"enabled": False},
            "vlm_yesno": {"cache_root": str(tmp_path)},
        }
    ).compute(_bundle())

    assert "vlm_insert_remove_margin" not in {
        metric.name for metric in without_vlm.metrics
    }
    vlm = next(
        metric
        for metric in with_vlm.metrics
        if metric.name == "vlm_insert_remove_margin"
    )
    assert vlm.metadata["available"] is False


def test_category_vlm_metric_uses_generic_name_and_validates_sidecar(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "insert_object"
    directory.mkdir()
    np.savez_compressed(
        directory / "episode_1.npz",
        center_sec=np.asarray([0.5, 1.5]),
        margin=np.asarray([-1.0, 2.0]),
        p_yes=np.asarray([0.25, 0.75]),
        yes_no_mass=np.asarray([0.9, 0.8]),
    )
    (directory / "episode_1.json").write_text(
        json.dumps(
            {
                "prompt": "fixed category prompt",
                "provenance": {"category": "acquire_release"},
            }
        )
    )
    config = {
        "surprise": {"enabled": False},
        "vlm_yesno": {
            "cache_root": str(tmp_path),
            "category": "acquire_release",
            "metric_name": "vlm_category_margin",
        },
    }

    profile = MetricEngine(config=config).compute(_bundle())

    metric = next(
        metric
        for metric in profile.metrics
        if metric.name == "vlm_category_margin"
    )
    assert metric.metadata["available"] is True
    assert metric.metadata["category"] == "acquire_release"
    assert metric.metadata["cache_provenance"]["category"] == "acquire_release"
