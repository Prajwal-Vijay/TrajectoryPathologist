"""Run REASSEMBLE leave-one-source-metric-out ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.config import load_pipeline_config
from pathologist.reassemble import run_reassemble_eval_pipeline


DEFAULT_SOURCE_METRICS = (
    "action_discontinuity",
    "motion_smoothness",
    "velocity_spike",
    "stalling",
    "eef_linear_velocity_spike",
    "eef_angular_velocity_spike",
    "eef_translational_stalling",
    "eef_rotational_stalling",
    "eef_translational_jerk",
    "eef_angular_jerk",
    "gripper_aperture_speed",
    "visual_flow",
    "audio_spectral_flux",
)
DEFAULT_COVERAGE_THRESHOLDS_PERCENT = (20.0, 30.0, 40.0, 50.0, 60.0)


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One source-metric configuration in an ablation sweep."""

    name: str
    source_metrics: tuple[str, ...]
    omitted_metric: str | None


def build_ablation_specs(
    source_metrics: list[str] | tuple[str, ...],
    mode: str = "leave-one-out",
) -> list[AblationSpec]:
    """Build a full baseline plus leave-one-out variants, or one custom run."""

    resolved = _resolve_source_metrics(source_metrics)
    if mode == "custom":
        return [AblationSpec(name="custom", source_metrics=resolved, omitted_metric=None)]
    if mode != "leave-one-out":
        raise ValueError("mode must be 'leave-one-out' or 'custom'.")
    if len(resolved) < 3:
        raise ValueError("Leave-one-out mode requires at least three source metrics.")
    return [
        AblationSpec(name="full", source_metrics=resolved, omitted_metric=None),
        *[
            AblationSpec(
                name=f"without_{_slug(metric_name)}",
                source_metrics=tuple(
                    candidate for candidate in resolved if candidate != metric_name
                ),
                omitted_metric=metric_name,
            )
            for metric_name in resolved
        ],
    ]


def run_reassemble_metric_ablation(
    *,
    h5_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path = "REASSEMBLE__metric_ablation_persistence",
    source_metrics: list[str] | tuple[str, ...] = DEFAULT_SOURCE_METRICS,
    mode: str = "leave-one-out",
    top_k: int = 5,
    config_path: str | Path | None = None,
    robot_downsample_hz: float | None = None,
    camera_names: list[str] | None = None,
    include_event_camera: bool | None = None,
    extract_frames: bool | None = None,
    frame_cache_dir: str | Path | None = None,
    max_files: int | None = None,
    coverage_thresholds_percent: list[float] | tuple[float, ...] = (
        DEFAULT_COVERAGE_THRESHOLDS_PERCENT
    ),
    plot_ecod_aggregates: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run persistence-ranked source-metric ablations and combine their results."""

    if h5_path is None and data_root is None:
        raise ValueError("Pass either h5_path or data_root.")
    if h5_path is not None and data_root is not None:
        raise ValueError("Pass only one of h5_path or data_root, not both.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    thresholds = _resolve_coverage_thresholds(coverage_thresholds_percent)
    specs = build_ablation_specs(source_metrics, mode=mode)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    shared_frame_cache = (
        Path(frame_cache_dir).expanduser().resolve()
        if frame_cache_dir is not None
        else output_path / "frame_cache"
    )
    shared_frame_cache.mkdir(parents=True, exist_ok=True)

    base_config = load_pipeline_config(config_path=config_path)
    run_summaries: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        run_dir = output_path / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        generated_config = _ablation_config(base_config, spec.source_metrics)
        generated_config_path = run_dir / "ablation_config.json"
        _write_json_atomic(generated_config_path, generated_config)
        metadata = {
            "ablation_name": spec.name,
            "omitted_metric": spec.omitted_metric,
            "source_metrics": list(spec.source_metrics),
            "metric_ranking": "persistence",
            "top_k": top_k,
        }
        metadata_path = run_dir / "ablation_metadata.json"
        existing_metadata = _read_json_object(metadata_path)
        existing_summary = _read_json_object(run_dir / "run_summary.json")
        can_reuse = (
            not force
            and existing_metadata == metadata
            and existing_summary is not None
            and existing_summary.get("status") == "complete"
        )
        if can_reuse:
            _log(verbose, f"[{index}/{len(specs)}] Reusing completed {spec.name}")
            run_summary = existing_summary
        else:
            _log(
                verbose,
                (
                    f"[{index}/{len(specs)}] Running {spec.name}: "
                    f"source_metrics={list(spec.source_metrics)}"
                ),
            )
            _write_json_atomic(metadata_path, metadata)
            run_summary = run_reassemble_eval_pipeline(
                h5_path=h5_path,
                data_root=data_root,
                output_dir=run_dir,
                top_k=top_k,
                segment_level="low",
                config_path=generated_config_path,
                robot_downsample_hz=robot_downsample_hz,
                camera_names=camera_names,
                include_event_camera=include_event_camera,
                extract_frames=extract_frames,
                frame_cache_dir=shared_frame_cache,
                max_files=max_files,
                selectors=["metric"],
                metric_ranking="persistence",
                plot_ecod_aggregates=plot_ecod_aggregates,
                verbose=verbose,
            )

        availability = _source_metric_availability(
            run_dir / "reassemble_alignment_debug.json",
            spec.source_metrics,
        )
        rows = _summarize_run(run_dir, spec, thresholds)
        result_rows.extend(rows)
        run_summaries.append(
            {
                **metadata,
                "output_dir": str(run_dir),
                "reused_completed_run": can_reuse,
                "source_metric_availability": availability,
                "pipeline_summary": run_summary,
            }
        )
        _write_combined_outputs(
            output_path=output_path,
            mode=mode,
            top_k=top_k,
            shared_frame_cache=shared_frame_cache,
            thresholds=thresholds,
            run_summaries=run_summaries,
            result_rows=result_rows,
            status="running" if index < len(specs) else "complete",
        )

    return _write_combined_outputs(
        output_path=output_path,
        mode=mode,
        top_k=top_k,
        shared_frame_cache=shared_frame_cache,
        thresholds=thresholds,
        run_summaries=run_summaries,
        result_rows=result_rows,
        status="complete",
    )


def _resolve_source_metrics(source_metrics: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    for raw_name in source_metrics:
        metric_name = str(raw_name).strip()
        if metric_name and metric_name not in resolved:
            resolved.append(metric_name)
    if len(resolved) < 2:
        raise ValueError("At least two unique source metrics are required for ECOD aggregation.")
    return tuple(resolved)


def _resolve_coverage_thresholds(values: list[float] | tuple[float, ...]) -> list[float]:
    thresholds = sorted({float(value) / 100.0 for value in values})
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Coverage thresholds must be percentages in the interval (0, 100].")
    return thresholds


def _ablation_config(
    base_config: dict[str, Any],
    source_metrics: tuple[str, ...],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    surprise_config = config.setdefault("surprise", {})
    if not isinstance(surprise_config, dict):
        raise ValueError("The surprise config must be an object.")
    surprise_config["source_metrics"] = list(source_metrics)
    selector_config = config.setdefault("selector", {})
    if not isinstance(selector_config, dict):
        raise ValueError("The selector config must be an object.")
    flood_config = selector_config.setdefault("surprise_flood", {})
    if not isinstance(flood_config, dict):
        raise ValueError("The selector.surprise_flood config must be an object.")
    flood_config["ranking"] = "persistence"
    return config


def _summarize_run(
    run_dir: Path,
    spec: AblationSpec,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "reassemble_manifest.json").read_text(encoding="utf-8"))
    eval_rows = [
        json.loads(line)
        for line in (run_dir / "reassemble_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coverage = {
        row["bundle_id"]: row["failed_low_level_coverage"]
        for row in manifest
        if row.get("included") and isinstance(row.get("failed_low_level_coverage"), dict)
    }
    rows_by_bundle: dict[str, dict[str, Any]] = {}
    for row in eval_rows:
        if row.get("selector_name") != "metric":
            continue
        bundle_id = str(row["bundle_id"])
        if bundle_id in rows_by_bundle:
            raise ValueError(f"Duplicate metric row in {run_dir}: {bundle_id}")
        rows_by_bundle[bundle_id] = row
    missing = sorted(set(coverage) - set(rows_by_bundle))
    if missing:
        raise ValueError(f"Missing metric rows in {run_dir}, first entries: {missing[:5]}")

    cohorts: list[tuple[str, float | None]] = [
        *[(f"<{threshold * 100:g}%", threshold) for threshold in thresholds],
        ("Full", None),
    ]
    summaries: list[dict[str, Any]] = []
    for cohort_name, threshold in cohorts:
        bundle_ids = sorted(
            bundle_id
            for bundle_id, values in coverage.items()
            if threshold is None
            or float(values["failed_low_level_coverage_fraction"]) < threshold
        )
        totals = {
            "trajectory_count": len(bundle_ids),
            "episode_hits": 0,
            "failed_segment_count": 0,
            "failed_segment_hits": 0,
            "selected_peak_count": 0,
            "selected_peak_hits": 0,
        }
        for bundle_id in bundle_ids:
            level = rows_by_bundle[bundle_id]["recall_by_level"]["low"]
            totals["episode_hits"] += int(bool(level["episode_hit_at_k"]))
            totals["failed_segment_count"] += int(level["failed_segment_count"])
            totals["failed_segment_hits"] += int(level["hit_failed_segment_count"])
            totals["selected_peak_count"] += int(level["selected_peak_count"])
            totals["selected_peak_hits"] += int(level["selected_peak_hit_count"])
        fractions = [
            float(coverage[bundle_id]["failed_low_level_coverage_fraction"])
            for bundle_id in bundle_ids
        ]
        recording_count = len(
            {str(rows_by_bundle[bundle_id]["h5_path"]) for bundle_id in bundle_ids}
        )
        summaries.append(
            {
                "ablation_name": spec.name,
                "omitted_metric": spec.omitted_metric,
                "source_metrics": list(spec.source_metrics),
                "cohort": cohort_name,
                "coverage_threshold_percentage": (
                    threshold * 100.0 if threshold is not None else None
                ),
                "trajectory_count": totals["trajectory_count"],
                "source_recording_count": recording_count,
                "mean_failed_low_level_fraction": (
                    sum(fractions) / len(fractions) if fractions else None
                ),
                **totals,
                "episode_recall_at_k": _ratio(
                    totals["episode_hits"], totals["trajectory_count"]
                ),
                "failed_segment_recall_at_k": _ratio(
                    totals["failed_segment_hits"], totals["failed_segment_count"]
                ),
                "point_precision_at_k": _ratio(
                    totals["selected_peak_hits"], totals["selected_peak_count"]
                ),
            }
        )
    return summaries


def _source_metric_availability(
    alignment_path: Path,
    source_metrics: tuple[str, ...],
) -> dict[str, Any]:
    entries = json.loads(alignment_path.read_text(encoding="utf-8"))
    included = [entry for entry in entries if entry.get("status") == "included"]
    unavailable_counts = {metric_name: 0 for metric_name in source_metrics}
    for entry in included:
        reasons = entry.get("metric_unavailable_reasons", {})
        if not isinstance(reasons, dict):
            continue
        for metric_name in source_metrics:
            if metric_name in reasons:
                unavailable_counts[metric_name] += 1
    return {
        "trajectory_count": len(included),
        "unavailable_trajectory_count_by_metric": unavailable_counts,
        "always_unavailable_metrics": [
            metric_name
            for metric_name, count in unavailable_counts.items()
            if included and count == len(included)
        ],
    }


def _write_combined_outputs(
    *,
    output_path: Path,
    mode: str,
    top_k: int,
    shared_frame_cache: Path,
    thresholds: list[float],
    run_summaries: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "dataset": "REASSEMBLE",
        "experiment": "source_metric_ablation",
        "mode": mode,
        "metric_ranking": "persistence",
        "top_k": top_k,
        "coverage_thresholds_percentage": [value * 100.0 for value in thresholds],
        "shared_frame_cache_dir": str(shared_frame_cache),
        "completed_ablation_count": len(run_summaries),
        "ablation_count": len(run_summaries),
        "runs": run_summaries,
        "results": result_rows,
    }
    _write_json_atomic(output_path / "ablation_summary.json", summary)
    _write_summary_csv(output_path / "ablation_summary.csv", result_rows)
    return summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ablation_name",
        "omitted_metric",
        "source_metrics",
        "cohort",
        "coverage_threshold_percentage",
        "trajectory_count",
        "source_recording_count",
        "mean_failed_low_level_fraction",
        "episode_hits",
        "episode_recall_at_k",
        "failed_segment_count",
        "failed_segment_hits",
        "failed_segment_recall_at_k",
        "selected_peak_count",
        "selected_peak_hits",
        "point_precision_at_k",
    ]
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["source_metrics"] = "|".join(row["source_metrics"])
            writer.writerow({field: serialized.get(field) for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "metric"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[reassemble-ablation] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run persistence-ranked REASSEMBLE source-metric ablations. "
            "Leave-one-out mode runs the full metric set and one run per omitted metric."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--h5-path", help="Path to one REASSEMBLE .h5 trajectory.")
    input_group.add_argument("--data-root", help="Directory containing REASSEMBLE .h5 trajectories.")
    parser.add_argument(
        "--output-dir",
        default="REASSEMBLE__metric_ablation_persistence",
        help="Root directory for all ablation runs and combined summaries.",
    )
    parser.add_argument(
        "--source-metrics",
        nargs="+",
        default=list(DEFAULT_SOURCE_METRICS),
        help="ECOD source metrics to use as the full set or exact custom set.",
    )
    parser.add_argument(
        "--mode",
        choices=["leave-one-out", "custom"],
        default="leave-one-out",
        help="Run a full leave-one-out sweep or one exact custom source-metric set.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--robot-downsample-hz", type=float, default=None)
    parser.add_argument(
        "--camera-name",
        "--camera",
        dest="camera_names",
        action="append",
        choices=["hand", "hama1", "hama2"],
        help="RGB camera to use. Can be passed multiple times; defaults to hand.",
    )
    parser.add_argument("--include-event-camera", action="store_true", default=None)
    parser.add_argument("--no-extract-frames", action="store_true")
    parser.add_argument(
        "--frame-cache-dir",
        default=None,
        help="Shared decoded-frame cache. Defaults to <output-dir>/frame_cache.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--coverage-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGE_THRESHOLDS_PERCENT),
        metavar="PERCENT",
        help="Failure-coverage percentages for combined summaries.",
    )
    parser.add_argument("--plot", action="store_true", help="Write ECOD plots for every ablation.")
    parser.add_argument("--force", action="store_true", help="Rerun completed matching ablations.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_reassemble_metric_ablation(
        h5_path=Path(args.h5_path) if args.h5_path is not None else None,
        data_root=Path(args.data_root) if args.data_root is not None else None,
        output_dir=args.output_dir,
        source_metrics=args.source_metrics,
        mode=args.mode,
        top_k=args.top_k,
        config_path=args.config_path,
        robot_downsample_hz=args.robot_downsample_hz,
        camera_names=args.camera_names,
        include_event_camera=args.include_event_camera,
        extract_frames=False if args.no_extract_frames else None,
        frame_cache_dir=args.frame_cache_dir,
        max_files=args.max_files,
        coverage_thresholds_percent=args.coverage_thresholds,
        plot_ecod_aggregates=args.plot,
        force=args.force,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
