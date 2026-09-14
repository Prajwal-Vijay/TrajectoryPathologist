"""Tests for the REASSEMBLE source-metric ablation runner."""

from __future__ import annotations

import pytest

from pathologist.run_reassemble_metric_ablation import (
    DEFAULT_SOURCE_METRICS,
    _ablation_config,
    _resolve_coverage_thresholds,
    build_ablation_specs,
    build_arg_parser,
)


def test_leave_one_out_specs_include_full_and_each_omission() -> None:
    specs = build_ablation_specs(DEFAULT_SOURCE_METRICS)

    assert specs[0].name == "full"
    assert specs[0].source_metrics == DEFAULT_SOURCE_METRICS
    assert len(specs) == len(DEFAULT_SOURCE_METRICS) + 1
    for spec in specs[1:]:
        assert spec.omitted_metric is not None
        assert spec.omitted_metric not in spec.source_metrics
        assert len(spec.source_metrics) == len(DEFAULT_SOURCE_METRICS) - 1


def test_custom_ablation_uses_exact_unique_metric_set() -> None:
    specs = build_ablation_specs(
        ["stalling", "visual_flow", "stalling", "velocity_spike"],
        mode="custom",
    )

    assert len(specs) == 1
    assert specs[0].source_metrics == ("stalling", "visual_flow", "velocity_spike")


def test_ablation_config_forces_persistence_and_source_metrics() -> None:
    config = _ablation_config(
        {"surprise": {}, "selector": {"surprise_flood": {"ranking": "area"}}},
        ("stalling", "visual_flow"),
    )

    assert config["surprise"]["source_metrics"] == ["stalling", "visual_flow"]
    assert config["selector"]["surprise_flood"]["ranking"] == "persistence"


def test_coverage_thresholds_are_percentages() -> None:
    assert _resolve_coverage_thresholds([50, 20, 50]) == [0.2, 0.5]
    with pytest.raises(ValueError, match="percentages"):
        _resolve_coverage_thresholds([0])


def test_ablation_cli_accepts_shared_cache_and_custom_mode() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data-root",
            "REASSEMBLE",
            "--mode",
            "custom",
            "--source-metrics",
            "stalling",
            "visual_flow",
            "--frame-cache-dir",
            "shared_frames",
        ]
    )

    assert args.mode == "custom"
    assert args.source_metrics == ["stalling", "visual_flow"]
    assert args.frame_cache_dir == "shared_frames"
