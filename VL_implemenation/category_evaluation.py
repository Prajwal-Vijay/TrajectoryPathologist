"""Evaluate VLM-only and telemetry+VLM persistence for ABC categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pathologist.adapters.abc130k_bundle import ABC130KBundleAdapter
from pathologist.events.metric_selector import MetricEventSelector
from pathologist.metrics import CachedVLMYesNoMetric, ECODSurpriseScorer
from pathologist.metrics.engine import default_metric_computers
from pathologist.schemas import CameraFrameRef
from pathologist.schemas.metrics import MetricProfile

from VL_implemenation.abc_data import (
    Episode,
    camera_cache_directory,
    camera_timeline,
    discover_episodes,
)
from VL_implemenation.category_protocol import (
    DEFAULT_EXPERIMENT_CONFIG,
    assignments_by_bundle,
    load_experiment_config,
    load_segment_assignments,
    selected_categories,
)
from VL_implemenation.coarse_to_fine import fine_cache_paths
from VL_implemenation.mosaic import CAMERA_NAMES
from VL_implemenation.prism_ablation import (
    _bootstrap_metrics,
    _paired_bootstrap,
    _score_rows,
)


VLM_METRIC_NAME = "vlm_category_margin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("VL_implemenation/config.json"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--robot-downsample-hz", type=float, default=50.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = load_experiment_config(args.experiment_config)
    categories = selected_categories(experiment, args.categories)
    episodes = discover_episodes(base_config["data_root"])
    if args.limit is not None:
        episodes = episodes[: args.limit]
    assignment_lookup = assignments_by_bundle(load_segment_assignments())
    bootstrap_samples = int(
        args.bootstrap_samples or experiment["evaluation"]["bootstrap_samples"]
    )
    output_root = (
        Path(base_config["output_root"]).expanduser().resolve()
        / "category_evaluation"
    )
    for category in categories:
        rows = run_category(
            category,
            episodes=episodes,
            assignments=assignment_lookup,
            experiment=experiment,
            frame_cache_root=Path(base_config["frame_cache_root"]),
            vlm_cache_root=(
                Path(base_config["output_root"]).expanduser().resolve()
                / "vlm_scores_by_category"
                / category
            ),
            fv_action_root=(
                Path(base_config["output_root"]).expanduser().resolve()
                / "fv_action_by_category"
                / category
            ),
            output_root=output_root / category,
            robot_downsample_hz=args.robot_downsample_hz,
            top_k=args.top_k,
            resume=args.resume,
        )
        summary = evaluate_category(
            category,
            rows=rows,
            output_root=output_root / category,
            bootstrap_samples=bootstrap_samples,
            seed=int(experiment["evaluation"]["seed"]),
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def run_category(
    category: str,
    *,
    episodes: Sequence[Episode],
    assignments: dict[str, list[dict[str, Any]]],
    experiment: dict[str, Any],
    frame_cache_root: Path,
    vlm_cache_root: Path,
    fv_action_root: Path,
    output_root: Path,
    robot_downsample_hz: float,
    top_k: int,
    resume: bool,
) -> list[dict[str, Any]]:
    """Compute exact configured metrics and persistence peaks for one category."""

    details = experiment["categories"][category]
    telemetry_names = [str(name) for name in details["telemetry_metrics"]]
    metrics = _metric_computers(telemetry_names)
    metrics.append(CachedVLMYesNoMetric(VLM_METRIC_NAME))
    base_metric_config = {
        "metrics": {
            "action_discontinuity": {"ignored_dimensions": []},
            "visual_flow": dict(experiment["visual_flow"]),
        },
        "surprise": {"enabled": False},
        "vlm_yesno": {
            "cache_root": str(vlm_cache_root),
            "category": category,
            "metric_name": VLM_METRIC_NAME,
        },
    }
    from pathologist.metrics import MetricEngine

    engine = MetricEngine(metrics=metrics, config=base_metric_config)
    method_configs = _method_configs(
        telemetry_names,
        experiment=experiment,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    peaks_path = output_root / "ranked_peaks.jsonl"
    existing = _read_jsonl(peaks_path) if resume else []
    completed = {str(row["bundle_id"]) for row in existing}
    if not resume and peaks_path.exists():
        peaks_path.unlink()
    rows = list(existing)
    adapter = ABC130KBundleAdapter()
    needs_visual = "visual_flow" in telemetry_names

    for index, episode in enumerate(episodes, start=1):
        if episode.bundle_id in completed:
            continue
        bundle = adapter.load(
            episode.episode_path,
            episode.annotation_path,
            extract_frames=False,
            robot_downsample_hz=robot_downsample_hz,
        )
        if needs_visual:
            bundle.camera_frames = _cached_camera_refs(
                episode,
                frame_cache_root=frame_cache_root,
                frame_rate=float(experiment["scan"]["frame_rate"]),
            )
        base_profile = engine.compute(bundle)
        metric_lookup = {metric.name: metric for metric in base_profile.metrics}
        requested_availability = {
            name: bool(metric_lookup[name].metadata.get("available", True))
            for name in [*telemetry_names, VLM_METRIC_NAME]
            if name in metric_lookup
        }
        missing = [
            name
            for name in [*telemetry_names, VLM_METRIC_NAME]
            if name not in metric_lookup or not requested_availability.get(name, False)
        ]
        method_peaks: dict[str, list[dict[str, float]]] = {}
        all_peaks: dict[str, list[dict[str, float]]] = {}
        method_sources: dict[str, list[str]] = {}
        for method_name, method_config in method_configs.items():
            profile = ECODSurpriseScorer(config=method_config).append_to_profile(
                MetricProfile(
                    bundle_id=base_profile.bundle_id,
                    metrics=list(base_profile.metrics),
                    global_events=list(base_profile.global_events),
                    metadata=dict(base_profile.metadata),
                )
            )
            proposals = _selector(method_config, 0).select(
                profile,
                trajectory_bundle=None,
            )
            serialized = [
                {
                    "rank": rank,
                    "time_sec": float(proposal.metadata["peak_sec"]),
                    "persistence": float(proposal.metadata["persistence"]),
                    "peak_score": float(proposal.metadata["peak_score"]),
                }
                for rank, proposal in enumerate(proposals, start=1)
            ]
            method_peaks[method_name] = serialized[:top_k]
            all_peaks[method_name] = serialized
            aggregate = next(
                (
                    metric
                    for metric in profile.metrics
                    if metric.name == "ecod_aggregate_score"
                ),
                None,
            )
            method_sources[method_name] = (
                list(aggregate.metadata.get("source_metric_names", []))
                if aggregate is not None
                else []
            )
        fine_path, _ = fine_cache_paths(
            fv_action_root,
            episode,
            subdirectory=None,
        )
        if fine_path.is_file():
            with np.load(fine_path, allow_pickle=False) as fine_cached:
                prediction_sec = float(fine_cached["prediction_sec"])
            method_peaks["fv_action"] = [
                {
                    "rank": 1,
                    "time_sec": prediction_sec,
                }
            ]
            all_peaks["fv_action"] = list(method_peaks["fv_action"])
            method_sources["fv_action"] = ["category_qwen_coarse_to_fine"]
        else:
            method_peaks["fv_action"] = []
            all_peaks["fv_action"] = []
            method_sources["fv_action"] = []
        ground_intervals = [
            {
                "event_type": category,
                "normalized_label": assignment["normalized_label"],
                "annotation_text": assignment["annotation_text"],
                "start_sec": float(assignment["start_sec"]),
                "end_sec": float(assignment["end_sec"]),
            }
            for assignment in assignments.get(episode.bundle_id, [])
            if assignment["category"] == category
        ]
        row = {
            "task_name": episode.task_name,
            "bundle_id": episode.bundle_id,
            "duration_sec": episode.duration_sec,
            "category": category,
            "ground_intervals": ground_intervals,
            "requested_metric_availability": requested_availability,
            "missing_requested_metrics": missing,
            "method_source_metrics": method_sources,
            "peaks": method_peaks,
            "all_peaks": all_peaks,
        }
        _append_jsonl(peaks_path, row)
        rows.append(row)
        print(
            f"[{category}] telemetry {index}/{len(episodes)} "
            f"{episode.task_name}/{episode.bundle_id}",
            flush=True,
        )
    _write_json_atomic(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "category": category,
            "episode_count": len(rows),
            "telemetry_metrics": telemetry_names,
            "vlm_metric": VLM_METRIC_NAME,
            "robot_downsample_hz": robot_downsample_hz,
            "top_k": top_k,
            "ranked_peaks_path": str(peaks_path),
            "missing_metric_episode_counts": {
                name: sum(name in row["missing_requested_metrics"] for row in rows)
                for name in [*telemetry_names, VLM_METRIC_NAME]
            },
            "fv_action_available_count": sum(
                bool(row["peaks"].get("fv_action")) for row in rows
            ),
        },
    )
    return rows


def evaluate_category(
    category: str,
    *,
    rows: Sequence[dict[str, Any]],
    output_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate category-positive episodes and paired persistence methods."""

    event_rows = [row for row in rows if row["ground_intervals"]]
    no_event_rows = [row for row in rows if not row["ground_intervals"]]
    table: dict[str, Any] = {}
    for method_name in ("vlm_only", "telemetry_vlm", "fv_action"):
        metrics = _score_rows(event_rows, method_name=method_name)
        metrics["bootstrap_95ci"] = _bootstrap_metrics(
            event_rows,
            method_name=method_name,
            samples=bootstrap_samples,
            seed=seed,
        )
        table[method_name] = metrics
    paired_fused = _paired_bootstrap(
        event_rows,
        left="telemetry_vlm",
        right="vlm_only",
        samples=bootstrap_samples,
        seed=seed,
    )
    paired_vs_fv = {
        method_name: _paired_bootstrap(
            event_rows,
            left=method_name,
            right="fv_action",
            samples=bootstrap_samples,
            seed=seed,
        )
        for method_name in ("vlm_only", "telemetry_vlm")
    }
    summary = {
        "protocol": {
            "category": category,
            "population": "episodes with at least one category segment",
            "interval_semantics": "half-open [start_sec, end_sec)",
            "persistence_methods": ["vlm_only", "telemetry_vlm"],
            "common_hit_at_1_methods": [
                "vlm_only",
                "telemetry_vlm",
                "fv_action",
            ],
            "bootstrap_episode_resamples": bootstrap_samples,
            "seed": seed,
        },
        "event_episode_count": len(event_rows),
        "ground_segment_count": sum(
            len(row["ground_intervals"]) for row in event_rows
        ),
        "no_event_episode_count": len(no_event_rows),
        "table": table,
        "paired_difference": {
            "telemetry_vlm_minus_vlm_only": paired_fused,
            "vlm_only_minus_fv_action": paired_vs_fv["vlm_only"],
            "telemetry_vlm_minus_fv_action": paired_vs_fv["telemetry_vlm"],
        },
    }
    _write_json_atomic(output_root / "evaluation.json", summary)
    return summary


def _method_configs(
    telemetry_names: list[str],
    *,
    experiment: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    persistence = dict(experiment["persistence"])
    aggregation = experiment["aggregation"]
    common = {
        "selector": {"surprise_flood": persistence},
    }
    return {
        "vlm_only": {
            **common,
            "surprise": {
                "source_metrics": [VLM_METRIC_NAME],
                "aggregation_reduction": "sum",
                "score_tail_overrides": {VLM_METRIC_NAME: "right"},
            },
        },
        "telemetry_vlm": {
            **common,
            "surprise": {
                "source_metrics": [*telemetry_names, VLM_METRIC_NAME],
                "aggregation_reduction": "sum",
                "score_tail_overrides": {VLM_METRIC_NAME: "right"},
                "aggregation_groups": {
                    "telemetry": telemetry_names,
                    "vlm": [VLM_METRIC_NAME],
                },
                "within_group_reduction": aggregation[
                    "within_telemetry_reduction"
                ],
                "across_group_reduction": aggregation[
                    "across_group_reduction"
                ],
            },
        },
    }


def _metric_computers(metric_names: Sequence[str]) -> list[Any]:
    registry = {metric.name: metric for metric in default_metric_computers()}
    unknown = sorted(set(metric_names) - set(registry))
    if unknown:
        raise ValueError(f"Requested metrics are not registered: {unknown}")
    return [registry[name] for name in metric_names]


def _cached_camera_refs(
    episode: Episode,
    *,
    frame_cache_root: Path,
    frame_rate: float,
) -> list[CameraFrameRef]:
    output: list[CameraFrameRef] = []
    for camera_name in CAMERA_NAMES:
        timeline = camera_timeline(
            camera_cache_directory(
                frame_cache_root,
                episode,
                camera_name,
                frame_rate,
            )
        )
        output.extend(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=float(timestamp),
                image_path=path,
                frame_index=index,
            )
            for index, (timestamp, path) in enumerate(timeline)
        )
    return output


def _selector(config: dict[str, Any], top_k: int) -> MetricEventSelector:
    flood = dict(config["selector"]["surprise_flood"])
    flood["max_proposals"] = top_k
    flood["ranking"] = "persistence"
    return MetricEventSelector(
        frame_stride_within_window=1,
        visual_context_sec=0.0,
        max_frames_per_camera=max(1, top_k),
        metric_packet_context_steps=0,
        surprise_flood_config=flood,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
