"""Exact-frame failure-localization evaluation for ViFailBack."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from pathologist.adapters.vifailback_bundle import (
    ViFailBackBundleAdapter,
    _annotation_keyframe_indices,
)
from pathologist.config import deep_merge, load_pipeline_config
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac_selectors import (
    select_kite_flow_keyframes,
    select_metric_keyframes,
    select_uniform_keyframes,
)
from pathologist.schemas import CameraFrameRef, TrajectoryBundle


NATIVE_SAMPLE_RATE_HZ = 25.0
DEFAULT_TOP_K = 2
DEFAULT_SMOOTHING_WINDOW = 11
DEFAULT_SELECTORS = ("random", "kite_flow", "metric")
SUPPORTED_SELECTORS = ("random", "uniform", "kite_flow", "metric")
DEFAULT_TOLERANCE_FRAMES = (0, 6, 12, 25)
VIFAILBACK_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True, slots=True)
class ViFailBackEpisodeRecord:
    """One failure episode with exact avoid-keyframe ground truth."""

    h5_path: Path
    annotation_path: Path
    task_name: str
    episode_stem: str
    avoid_keyframe_indices: tuple[int, ...]
    failure_type: str | None
    failure_subtask: str | None

    @property
    def bundle_id(self) -> str:
        return f"{self.task_name}_{self.episode_stem}"


def discover_vifailback_failure_episodes(
    data_root: str | Path,
) -> list[ViFailBackEpisodeRecord]:
    """Discover downloaded failure episodes with exact avoid-keyframe labels."""

    root = Path(data_root).expanduser().resolve()
    raw_root = root / "raw_data"
    annotation_root = root / "annotated_data"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Could not find ViFailBack raw_data directory: {raw_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Could not find ViFailBack annotated_data directory: {annotation_root}"
        )

    records: list[ViFailBackEpisodeRecord] = []
    for annotation_path in sorted(
        annotation_root.glob("*/*_annotations.json")
    ):
        task_name = annotation_path.parent.name
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of annotations in {annotation_path}")
        for annotation in payload:
            if not isinstance(annotation, dict):
                continue
            if str(annotation.get("failure_detection", "")).strip().lower() != "yes":
                continue
            episode_stem = _episode_stem_from_annotation(annotation)
            avoid_indices = _annotation_keyframe_indices(annotation, "avoid")
            if episode_stem is None or not avoid_indices:
                continue
            h5_path = raw_root / task_name / f"{episode_stem}.hdf5"
            if not h5_path.is_file():
                continue
            records.append(
                ViFailBackEpisodeRecord(
                    h5_path=h5_path,
                    annotation_path=annotation_path,
                    task_name=task_name,
                    episode_stem=episode_stem,
                    avoid_keyframe_indices=tuple(avoid_indices),
                    failure_type=_optional_string(annotation.get("failure_type")),
                    failure_subtask=_optional_string(annotation.get("failure_subtask")),
                )
            )
    return sorted(records, key=lambda item: (item.task_name, _natural_episode_key(item.episode_stem)))


def run_vifailback_localization_eval(
    *,
    data_root: str | Path = "/media/helen/28TB_VOL4/ViFailback-Dataset",
    output_dir: str | Path = "ViFailBack_localization_k2",
    top_k: int = DEFAULT_TOP_K,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    selectors: list[str] | tuple[str, ...] | None = None,
    tolerance_frames: list[int] | tuple[int, ...] = DEFAULT_TOLERANCE_FRAMES,
    max_episodes: int | None = None,
    config_path: str | Path | None = None,
    frame_cache_dir: str | Path | None = None,
    keep_frame_cache: bool = False,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    resume: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compare fixed-budget selectors against exact 25-Hz avoid keyframes."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive when provided.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative.")
    resolved_selectors = _resolve_selectors(selectors)
    resolved_tolerances = sorted({int(value) for value in tolerance_frames})
    if not resolved_tolerances or resolved_tolerances[0] < 0:
        raise ValueError("tolerance_frames must contain non-negative frame counts.")

    root = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = (
        Path(frame_cache_dir).expanduser().resolve()
        if frame_cache_dir is not None
        else output / "frame_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    eval_path = output / "vifailback_localization_eval.jsonl"
    manifest_path = output / "vifailback_manifest.json"
    stats_path = output / "vifailback_localization_stats.json"
    summary_path = output / "run_summary.json"

    records = discover_vifailback_failure_episodes(root)
    discovered_failure_episode_count = len(records)
    if max_episodes is not None:
        records = records[:max_episodes]
    config = deep_merge(
        load_pipeline_config(config_path=config_path),
        _vifailback_config_override(
            top_k,
            smoothing_window=smoothing_window,
        ),
    )

    existing_rows = _read_jsonl(eval_path) if resume else []
    requested_selectors = set(resolved_selectors)
    completed_by_bundle: dict[str, set[str]] = defaultdict(set)
    for row in existing_rows:
        completed_by_bundle[str(row.get("bundle_id", ""))].add(
            str(row.get("selector_name", ""))
        )
    completed = {
        bundle_id
        for bundle_id, names in completed_by_bundle.items()
        if requested_selectors.issubset(names)
    }
    rows = [
        row
        for row in existing_rows
        if str(row.get("bundle_id", "")) in completed
        and str(row.get("selector_name", "")) in requested_selectors
    ]
    manifest = _read_json(manifest_path, default=[]) if resume else []
    if not isinstance(manifest, list):
        manifest = []
    manifest = [
        entry
        for entry in manifest
        if isinstance(entry, dict) and str(entry.get("bundle_id", "")) in completed
    ]
    failure_type_by_bundle = {
        str(entry["bundle_id"]): str(entry["failure_type"])
        for entry in manifest
        if entry.get("bundle_id") and entry.get("failure_type")
    }
    rows_enriched = False
    for row in rows:
        if not row.get("failure_type"):
            failure_type = failure_type_by_bundle.get(str(row.get("bundle_id", "")))
            if failure_type is not None:
                row["failure_type"] = failure_type
                rows_enriched = True
    if not resume:
        eval_path.write_text("", encoding="utf-8")
    elif rows_enriched:
        _write_jsonl(eval_path, rows)

    summary_base = {
        "status": "running",
        "data_root": str(root),
        "output_dir": str(output),
        "top_k": top_k,
        "smoothing_window": smoothing_window,
        "selectors": resolved_selectors,
        "sample_rate_hz": NATIVE_SAMPLE_RATE_HZ,
        "ground_truth": "avoid_keyframe_exact_source_frame",
        "tolerance_frames": resolved_tolerances,
        "tolerance_seconds": [
            value / NATIVE_SAMPLE_RATE_HZ for value in resolved_tolerances
        ],
        "discovered_failure_episode_count": discovered_failure_episode_count,
        "selected_failure_episode_count": len(records),
        "frame_cache_dir": str(cache_root),
        "keep_frame_cache": keep_frame_cache,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "eval_path": str(eval_path),
        "manifest_path": str(manifest_path),
        "stats_path": str(stats_path),
    }
    _write_json(summary_path, summary_base)

    adapter = ViFailBackBundleAdapter()
    for index, record in enumerate(records, start=1):
        if record.bundle_id in completed:
            _log(verbose, f"[{index}/{len(records)}] resume skip {record.bundle_id}")
            continue
        _log(verbose, f"[{index}/{len(records)}] {record.bundle_id}")
        episode_cache = cache_root / record.task_name / record.episode_stem
        try:
            bundle = adapter.load(
                record.h5_path,
                frame_cache_dir=episode_cache,
                hdf5_fps=NATIVE_SAMPLE_RATE_HZ,
                default_control_hz=NATIVE_SAMPLE_RATE_HZ,
            )
            if tuple(bundle.raw_metadata.get("avoid_keyframe_indices", [])) != (
                record.avoid_keyframe_indices
            ):
                raise ValueError(
                    "Adapter annotation metadata does not match discovery metadata: "
                    f"{bundle.raw_metadata.get('avoid_keyframe_indices')} != "
                    f"{list(record.avoid_keyframe_indices)}"
                )
            metric_profile = (
                MetricEngine(config=config).compute(bundle)
                if "metric" in resolved_selectors
                else None
            )
            episode_rows = [
                _evaluate_selector(
                    selector_name=selector_name,
                    bundle=bundle,
                    top_k=top_k,
                    ground_truth_frames=list(record.avoid_keyframe_indices),
                    tolerance_frames=resolved_tolerances,
                    config=config,
                    metric_profile=metric_profile,
                    seed=seed,
                )
                for selector_name in resolved_selectors
            ]
            _append_jsonl(eval_path, episode_rows)
            rows.extend(episode_rows)
            manifest.append(
                {
                    "bundle_id": record.bundle_id,
                    "task_name": record.task_name,
                    "h5_path": str(record.h5_path),
                    "annotation_path": str(record.annotation_path),
                    "sample_count": bundle.raw_metadata.get("sample_count"),
                    "duration_sec": bundle.duration_sec,
                    "avoid_keyframe_indices": list(record.avoid_keyframe_indices),
                    "failure_type": record.failure_type,
                    "failure_subtask": record.failure_subtask,
                    "metric_availability": (
                        _metric_availability(metric_profile)
                        if metric_profile is not None
                        else {}
                    ),
                    "status": "complete",
                }
            )
        except Exception as exc:
            manifest.append(
                {
                    "bundle_id": record.bundle_id,
                    "task_name": record.task_name,
                    "h5_path": str(record.h5_path),
                    "annotation_path": str(record.annotation_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(verbose, f"  ERROR {type(exc).__name__}: {exc}")
        finally:
            if not keep_frame_cache:
                shutil.rmtree(episode_cache, ignore_errors=True)

        _write_json(manifest_path, manifest)
        _write_json(
            stats_path,
            aggregate_vifailback_localization(
                rows,
                tolerance_frames=resolved_tolerances,
                bootstrap_samples=0,
                seed=seed,
            ),
        )
        _write_json(
            summary_path,
            {
                **summary_base,
                "processed_episode_count": sum(
                    entry.get("status") == "complete" for entry in manifest
                ),
                "error_count": sum(
                    entry.get("status") == "error" for entry in manifest
                ),
            },
        )

    stats = aggregate_vifailback_localization(
        rows,
        tolerance_frames=resolved_tolerances,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    _write_json(manifest_path, manifest)
    _write_json(stats_path, stats)
    summary = {
        **summary_base,
        "status": "complete",
        "processed_episode_count": sum(
            entry.get("status") == "complete" for entry in manifest
        ),
        "error_count": sum(entry.get("status") == "error" for entry in manifest),
        "evaluation_row_count": len(rows),
        "stats": stats,
    }
    _write_json(summary_path, summary)
    return summary


def evaluate_exact_frame_predictions(
    *,
    ground_truth_frames: list[int],
    predicted_frames: list[int],
    tolerance_frames: list[int] | tuple[int, ...] = DEFAULT_TOLERANCE_FRAMES,
) -> dict[str, Any]:
    """Score predicted source-frame indices without temporal resampling."""

    targets = sorted({int(value) for value in ground_truth_frames})
    predictions = [int(value) for value in predicted_frames]
    tolerances = sorted({int(value) for value in tolerance_frames})
    if not targets:
        raise ValueError("At least one ground-truth frame is required.")
    if any(value < 0 for value in (*targets, *predictions, *tolerances)):
        raise ValueError("Frame indices and tolerances must be non-negative.")

    ground_truth_matches = []
    for target in targets:
        nearest_prediction = (
            min(predictions, key=lambda value: (abs(value - target), value))
            if predictions
            else None
        )
        signed_error = (
            nearest_prediction - target if nearest_prediction is not None else None
        )
        ground_truth_matches.append(
            {
                "ground_truth_frame_index": target,
                "nearest_predicted_frame_index": nearest_prediction,
                "signed_frame_error": signed_error,
                "absolute_frame_error": (
                    abs(signed_error) if signed_error is not None else None
                ),
            }
        )

    prediction_matches = []
    for prediction in predictions:
        nearest_target = min(targets, key=lambda value: (abs(value - prediction), value))
        signed_error = prediction - nearest_target
        prediction_matches.append(
            {
                "predicted_frame_index": prediction,
                "nearest_ground_truth_frame_index": nearest_target,
                "signed_frame_error": signed_error,
                "absolute_frame_error": abs(signed_error),
            }
        )

    finite_errors = [
        int(match["absolute_frame_error"])
        for match in ground_truth_matches
        if match["absolute_frame_error"] is not None
    ]
    return {
        "ground_truth_matches": ground_truth_matches,
        "prediction_matches": prediction_matches,
        "nearest_absolute_frame_error": min(finite_errors) if finite_errors else None,
        "nearest_absolute_error_sec": (
            min(finite_errors) / NATIVE_SAMPLE_RATE_HZ if finite_errors else None
        ),
        "ground_truth_hits_by_tolerance_frames": {
            str(tolerance): sum(error <= tolerance for error in finite_errors)
            for tolerance in tolerances
        },
        "prediction_hits_by_tolerance_frames": {
            str(tolerance): sum(
                int(match["absolute_frame_error"]) <= tolerance
                for match in prediction_matches
            )
            for tolerance in tolerances
        },
    }


def aggregate_vifailback_localization(
    rows: list[dict[str, Any]],
    *,
    tolerance_frames: list[int] | tuple[int, ...] = DEFAULT_TOLERANCE_FRAMES,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate exact-frame localization error and hit rates by selector."""

    tolerances = sorted({int(value) for value in tolerance_frames})
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_selector[str(row["selector_name"])].append(row)

    output: dict[str, Any] = {}
    for selector_name, selector_rows in sorted(by_selector.items()):
        point = _aggregate_selector_rows(selector_rows, tolerances)
        point["episode_clustered_bootstrap_95ci"] = (
            _bootstrap_selector_rows(
                selector_rows,
                tolerances=tolerances,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            if bootstrap_samples
            else {}
        )
        output[selector_name] = point
    rows_by_failure_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        failure_type = row.get("failure_type")
        if isinstance(failure_type, str) and failure_type:
            rows_by_failure_type[failure_type].append(row)
    output["by_failure_type"] = {
        failure_type: {
            selector_name: _aggregate_selector_rows(selector_rows, tolerances)
            for selector_name, selector_rows in sorted(
                _rows_by_selector(failure_rows).items()
            )
        }
        for failure_type, failure_rows in sorted(rows_by_failure_type.items())
    }
    return output


def _evaluate_selector(
    *,
    selector_name: str,
    bundle: TrajectoryBundle,
    top_k: int,
    ground_truth_frames: list[int],
    tolerance_frames: list[int],
    config: dict[str, Any],
    metric_profile: Any | None,
    seed: int,
) -> dict[str, Any]:
    selection_stats: dict[str, Any] = {}
    if selector_name == "random":
        high_frames = _camera_frames(bundle.camera_frames, "cam_high")
        episode_seed = f"{seed}:{bundle.bundle_id}:random"
        frames = _select_random_keyframes(
            high_frames,
            top_k,
            seed=episode_seed,
        )
        selection_stats = {
            "protocol": "seeded_random_without_replacement_over_native_cam_high_frames",
            "seed": episode_seed,
            "method_frame_count": len(frames),
            "frame_budget": top_k,
        }
    elif selector_name == "uniform":
        high_frames = _camera_frames(bundle.camera_frames, "cam_high")
        frames = select_uniform_keyframes(high_frames, top_k)
        selection_stats = {
            "protocol": "uniform_over_native_cam_high_frames",
            "method_frame_count": len(frames),
            "frame_budget": top_k,
        }
    elif selector_name == "kite_flow":
        frames = select_kite_flow_keyframes(
            bundle.camera_frames,
            top_k,
            config=config,
            selection_stats=selection_stats,
            pad_to_budget=True,
        )
    elif selector_name == "metric":
        if metric_profile is None:
            raise ValueError("metric selector requires a metric profile.")
        frames = select_metric_keyframes(
            bundle,
            metric_profile,
            top_k,
            config=config,
            selection_stats=selection_stats,
        )
    else:
        raise ValueError(f"Unsupported ViFailBack selector: {selector_name}")

    predictions = [
        {
            "rank": rank,
            "frame_index": _exact_frame_index(frame),
            "timestamp_sec": frame.timestamp_sec,
            "camera_name": frame.camera_name,
        }
        for rank, frame in enumerate(frames[:top_k], start=1)
    ]
    predicted_frames = [int(item["frame_index"]) for item in predictions]
    scoring = evaluate_exact_frame_predictions(
        ground_truth_frames=ground_truth_frames,
        predicted_frames=predicted_frames,
        tolerance_frames=tolerance_frames,
    )
    return {
        "bundle_id": bundle.bundle_id,
        "task_name": bundle.raw_metadata.get("source_task"),
        "failure_type": bundle.raw_metadata.get("failure_type"),
        "failure_subtask": bundle.raw_metadata.get("failure_subtask"),
        "h5_path": str(bundle.root_path),
        "selector_name": selector_name,
        "top_k": top_k,
        "sample_rate_hz": NATIVE_SAMPLE_RATE_HZ,
        "sample_count": bundle.raw_metadata.get("sample_count"),
        "duration_sec": bundle.duration_sec,
        "ground_truth_frame_indices": ground_truth_frames,
        "ground_truth_timestamps_sec": [
            value / NATIVE_SAMPLE_RATE_HZ for value in ground_truth_frames
        ],
        "selected_predictions": predictions,
        "selected_prediction_count": len(predictions),
        "selection_protocol": selection_stats,
        **scoring,
    }


def _aggregate_selector_rows(
    rows: list[dict[str, Any]],
    tolerances: list[int],
) -> dict[str, Any]:
    target_errors = [
        int(match["absolute_frame_error"])
        for row in rows
        for match in row.get("ground_truth_matches", [])
        if match.get("absolute_frame_error") is not None
    ]
    prediction_errors = [
        int(match["absolute_frame_error"])
        for row in rows
        for match in row.get("prediction_matches", [])
        if match.get("absolute_frame_error") is not None
    ]
    target_count = sum(
        len(row.get("ground_truth_frame_indices", [])) for row in rows
    )
    prediction_count = sum(
        int(row.get("selected_prediction_count", 0)) for row in rows
    )
    target_hits = {
        tolerance: sum(error <= tolerance for error in target_errors)
        for tolerance in tolerances
    }
    prediction_hits = {
        tolerance: sum(error <= tolerance for error in prediction_errors)
        for tolerance in tolerances
    }
    return {
        "trajectory_count": len(rows),
        "ground_truth_keyframe_count": target_count,
        "selected_prediction_count": prediction_count,
        "mean_nearest_absolute_frame_error": (
            sum(target_errors) / len(target_errors) if target_errors else None
        ),
        "median_nearest_absolute_frame_error": (
            median(target_errors) if target_errors else None
        ),
        "mean_nearest_absolute_error_sec": (
            sum(target_errors) / len(target_errors) / NATIVE_SAMPLE_RATE_HZ
            if target_errors
            else None
        ),
        "keyframe_recall_at_k_by_tolerance_frames": {
            str(tolerance): (
                target_hits[tolerance] / target_count if target_count else None
            )
            for tolerance in tolerances
        },
        "point_precision_at_k_by_tolerance_frames": {
            str(tolerance): (
                prediction_hits[tolerance] / prediction_count
                if prediction_count
                else None
            )
            for tolerance in tolerances
        },
    }


def _rows_by_selector(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["selector_name"])].append(row)
    return output


def _bootstrap_selector_rows(
    rows: list[dict[str, Any]],
    *,
    tolerances: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_samples):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        aggregate = _aggregate_selector_rows(sampled, tolerances)
        mean_error = aggregate.get("mean_nearest_absolute_frame_error")
        if mean_error is not None:
            samples["mean_nearest_absolute_frame_error"].append(float(mean_error))
        recall = aggregate["keyframe_recall_at_k_by_tolerance_frames"]
        for tolerance in tolerances:
            value = recall.get(str(tolerance))
            if value is not None:
                samples[f"keyframe_recall_frames_{tolerance}"].append(float(value))
    return {
        key: _percentile_interval(values)
        for key, values in sorted(samples.items())
    }


def _vifailback_config_override(
    top_k: int,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> dict[str, Any]:
    camera_names = list(VIFAILBACK_CAMERA_NAMES)
    return {
        "metrics": {
            "visual_flow": {
                "camera_names": camera_names,
                "spatial_aggregation": "top_fraction_mean",
                "top_fraction": 0.02,
            },
            "visual_patch_novelty": {"camera_names": camera_names},
            "object_interaction_motion": {"camera_names": camera_names},
        },
        "robofac_qa": {
            "kite_flow": {
                "camera_name": "cam_high",
            }
        },
        "selector": {
            "surprise_flood": {
                "smoothing_window": smoothing_window,
                "ranking": "persistence",
                "max_proposals": top_k,
            }
        },
    }


def _resolve_selectors(
    selectors: list[str] | tuple[str, ...] | None,
) -> list[str]:
    aliases = {"kite": "kite_flow", "flow": "kite_flow", "prism": "metric"}
    output: list[str] = []
    for name in selectors or DEFAULT_SELECTORS:
        resolved = aliases.get(str(name).strip().lower(), str(name).strip().lower())
        if resolved not in SUPPORTED_SELECTORS:
            raise ValueError(f"Unsupported ViFailBack selector: {name}")
        if resolved not in output:
            output.append(resolved)
    return output


def _select_random_keyframes(
    frames: list[CameraFrameRef],
    frame_budget: int,
    *,
    seed: str | int,
) -> list[CameraFrameRef]:
    """Select reproducible native frames uniformly without replacement."""

    ordered = sorted(
        frames,
        key=lambda frame: (
            frame.frame_index if frame.frame_index is not None else -1,
            frame.timestamp_sec,
            frame.camera_name,
        ),
    )
    if len(ordered) <= frame_budget:
        return ordered
    selected = random.Random(seed).sample(ordered, frame_budget)
    return sorted(
        selected,
        key=lambda frame: (
            frame.frame_index if frame.frame_index is not None else -1,
            frame.timestamp_sec,
            frame.camera_name,
        ),
    )


def _episode_stem_from_annotation(annotation: dict[str, Any]) -> str | None:
    video = annotation.get("video")
    if not isinstance(video, str):
        return None
    stem = Path(video).stem
    suffix = "_cam_high"
    return stem[: -len(suffix)] if stem.endswith(suffix) else None


def _camera_frames(
    frames: list[CameraFrameRef],
    camera_name: str,
) -> list[CameraFrameRef]:
    return sorted(
        (frame for frame in frames if frame.camera_name == camera_name),
        key=lambda frame: (
            frame.frame_index if frame.frame_index is not None else -1,
            frame.timestamp_sec,
        ),
    )


def _exact_frame_index(frame: CameraFrameRef) -> int:
    if frame.frame_index is not None:
        expected_timestamp = frame.frame_index / NATIVE_SAMPLE_RATE_HZ
        if abs(frame.timestamp_sec - expected_timestamp) > 1e-9:
            raise ValueError(
                "ViFailBack frame timestamp/index mismatch: "
                f"index={frame.frame_index}, timestamp={frame.timestamp_sec}"
            )
        return int(frame.frame_index)
    index = round(frame.timestamp_sec * NATIVE_SAMPLE_RATE_HZ)
    if abs(frame.timestamp_sec - index / NATIVE_SAMPLE_RATE_HZ) > 1e-9:
        raise ValueError(
            f"Timestamp {frame.timestamp_sec} is not on the native 25-Hz frame grid."
        )
    return index


def _metric_availability(profile: Any) -> dict[str, bool]:
    return {
        metric.name: bool(metric.metadata.get("available", bool(metric.samples)))
        for metric in profile.metrics
    }


def _natural_episode_key(stem: str) -> tuple[str, int]:
    prefix, separator, suffix = stem.rpartition("_")
    return (prefix if separator else stem, int(suffix) if suffix.isdigit() else -1)


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return [
        ordered[round(0.025 * (len(ordered) - 1))],
        ordered[round(0.975 * (len(ordered) - 1))],
    ]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)
