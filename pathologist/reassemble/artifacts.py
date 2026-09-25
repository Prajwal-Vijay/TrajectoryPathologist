"""Per-trajectory exports for REASSEMBLE salient-event labeling."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.metrics import MetricProfile, MetricResult
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


DEFAULT_PEAK_CAMERAS = ("hama1", "hama2", "hand")


def export_per_trajectory_results(
    *,
    results_root: str | Path,
    bundle: TrajectoryBundle,
    metric_profile: MetricProfile,
    proposals: list[KeyEventProposal],
    aggregate_debug_path: str | Path,
    ground_truth_events: Iterable[Any] = (),
    annotation_labels: Iterable[str] = (),
    camera_names: Iterable[str] = DEFAULT_PEAK_CAMERAS,
) -> dict[str, Any]:
    """Write ECOD series, peak frames, annotations, and a combined plot."""

    trajectory_dir = Path(results_root).expanduser().resolve() / _safe_name(bundle.bundle_id)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    aggregate_debug = _read_json(Path(aggregate_debug_path))
    aggregate_metric = _find_aggregate_metric(metric_profile)
    if aggregate_metric is None:
        raise ValueError(f"No available aggregate ECOD metric for {bundle.bundle_id}")

    source_names = [
        str(name)
        for name in aggregate_metric.metadata.get("source_metric_names", [])
        if isinstance(name, str)
    ]
    ecod_by_source = {
        str(metric.metadata.get("source_metric_name")): metric
        for metric in metric_profile.metrics
        if metric.name.endswith("_ecod_score")
        and isinstance(metric.metadata.get("source_metric_name"), str)
    }

    metric_files: dict[str, str] = {}
    for source_name in source_names:
        ecod_metric = ecod_by_source.get(source_name)
        if ecod_metric is None:
            continue
        destination = trajectory_dir / f"ECOD_{_safe_name(source_name)}.json"
        _write_json(
            destination,
            _metric_series_payload(
                bundle_id=bundle.bundle_id,
                source_metric_name=source_name,
                metric=ecod_metric,
            ),
        )
        metric_files[source_name] = str(destination)

    aggregate_path = trajectory_dir / "ECOD_aggregate_score.json"
    aggregate_payload = {
        "bundle_id": bundle.bundle_id,
        "metric_name": aggregate_metric.name,
        "aggregation_reduction": aggregate_metric.metadata.get("aggregation_reduction"),
        "source_metric_names": source_names,
        "timestamps_sec": [float(value) for value in aggregate_debug.get("timestamps_sec", [])],
        "ecod_scores": [float(value) for value in aggregate_debug.get("raw_scores", [])],
        "smoothed_ecod_scores": [
            float(value) for value in aggregate_debug.get("smoothed_scores", [])
        ],
        "smoothing": aggregate_debug.get("smoothing"),
    }
    _write_json(aggregate_path, aggregate_payload)

    events = [_event_to_dict(event) for event in ground_truth_events]
    annotations_path = trajectory_dir / "ground_truth_annotations.json"
    _write_json(
        annotations_path,
        {
            "bundle_id": bundle.bundle_id,
            "included_labels": list(annotation_labels),
            "events": events,
        },
    )

    peaks = _export_peak_frames(
        trajectory_dir=trajectory_dir,
        bundle=bundle,
        proposals=proposals,
        camera_names=tuple(camera_names),
    )
    peaks_path = trajectory_dir / "selected_peaks.json"
    _write_json(peaks_path, {"bundle_id": bundle.bundle_id, "peaks": peaks})

    plot_path = trajectory_dir / "aggregate_ecod_peaks_ground_truth.png"
    plot_error = _write_combined_plot(
        path=plot_path,
        bundle_id=bundle.bundle_id,
        aggregate_payload=aggregate_payload,
        peaks=peaks,
        events=events,
        annotation_labels=list(annotation_labels),
    )
    if plot_error is not None:
        raise RuntimeError(f"Could not plot {bundle.bundle_id}: {plot_error}")

    summary = {
        "bundle_id": bundle.bundle_id,
        "trajectory_dir": str(trajectory_dir),
        "ecod_metric_count": len(metric_files),
        "ecod_metric_files": metric_files,
        "aggregate_file": str(aggregate_path),
        "selected_peaks_file": str(peaks_path),
        "ground_truth_file": str(annotations_path),
        "plot_file": str(plot_path),
        "selected_peak_count": len(peaks),
        "ground_truth_event_count": len(events),
        "camera_names": list(camera_names),
    }
    _write_json(trajectory_dir / "export_summary.json", summary)
    return summary


def _find_aggregate_metric(profile: MetricProfile) -> MetricResult | None:
    for metric in profile.metrics:
        if metric.display_hint == "ecod_aggregate_surprise_timeseries" and metric.samples:
            return metric
    return None


def _metric_series_payload(
    *,
    bundle_id: str,
    source_metric_name: str,
    metric: MetricResult,
) -> dict[str, Any]:
    samples = [
        sample
        for sample in metric.samples
        if isinstance(sample.value, (int, float, bool))
    ]
    return {
        "bundle_id": bundle_id,
        "source_metric_name": source_metric_name,
        "ecod_metric_name": metric.name,
        "timestamps_sec": [float(sample.timestamp_sec) for sample in samples],
        "ecod_scores": [float(sample.value) for sample in samples],
        "score_tail_mode": metric.metadata.get("score_tail_mode"),
        "source_unit": metric.metadata.get("source_unit"),
    }


def _export_peak_frames(
    *,
    trajectory_dir: Path,
    bundle: TrajectoryBundle,
    proposals: list[KeyEventProposal],
    camera_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    frames_by_camera: dict[str, list[CameraFrameRef]] = {
        camera_name: sorted(
            (frame for frame in bundle.camera_frames if frame.camera_name == camera_name),
            key=lambda frame: frame.timestamp_sec,
        )
        for camera_name in camera_names
    }
    peak_frames_root = trajectory_dir / "peak frames"
    for camera_name in camera_names:
        (peak_frames_root / _safe_name(camera_name)).mkdir(parents=True, exist_ok=True)

    output: list[dict[str, Any]] = []
    for rank, proposal in enumerate(proposals, start=1):
        peak_sec = _proposal_peak_sec(proposal)
        peak_payload: dict[str, Any] = {
            "rank": rank,
            "proposal_id": proposal.event_id,
            "peak_sec": peak_sec,
            "frames": {},
        }
        for camera_name in camera_names:
            frames = frames_by_camera[camera_name]
            if not frames:
                peak_payload["frames"][camera_name] = {"available": False}
                continue
            frame = min(frames, key=lambda item: abs(item.timestamp_sec - peak_sec))
            suffix = frame.image_path.suffix.lower() or ".jpg"
            target = (
                peak_frames_root
                / _safe_name(camera_name)
                / f"peak_{rank:02d}_t{peak_sec:08.3f}_frame_{frame.frame_index or 0:06d}{suffix}"
            )
            if not target.exists():
                shutil.copy2(frame.image_path, target)
            peak_payload["frames"][camera_name] = {
                "available": True,
                "frame_index": frame.frame_index,
                "frame_timestamp_sec": float(frame.timestamp_sec),
                "offset_from_peak_sec": float(frame.timestamp_sec - peak_sec),
                "image_path": str(target),
            }
        output.append(peak_payload)
    return output


def _proposal_peak_sec(proposal: KeyEventProposal) -> float:
    value = proposal.metadata.get("peak_sec")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    end_sec = proposal.time_window.end_sec
    if end_sec is None:
        return float(proposal.time_window.start_sec)
    return float((proposal.time_window.start_sec + end_sec) / 2.0)


def _event_to_dict(event: Any) -> dict[str, Any]:
    if hasattr(event, "to_dict"):
        payload = event.to_dict()
    elif isinstance(event, dict):
        payload = dict(event)
    else:
        raise TypeError(f"Unsupported ground-truth event type: {type(event).__name__}")
    return {
        "label": str(payload["label"]),
        "start_sec": float(payload["start_sec"]),
        "end_sec": float(payload["end_sec"]),
        "reference_sec": float(payload["reference_sec"]),
        "raw_line": payload.get("raw_line"),
    }


def _write_combined_plot(
    *,
    path: Path,
    bundle_id: str,
    aggregate_payload: dict[str, Any],
    peaks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    annotation_labels: list[str],
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathologist_matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pathologist_cache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    try:
        timestamps = aggregate_payload["timestamps_sec"]
        raw_scores = aggregate_payload["ecod_scores"]
        smoothed_scores = aggregate_payload["smoothed_ecod_scores"]
        figure, (score_axis, annotation_axis) = plt.subplots(
            2,
            1,
            figsize=(13, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3.5, 1.6]},
        )
        score_axis.plot(
            timestamps,
            raw_scores,
            color="#9ca3af",
            linewidth=0.8,
            alpha=0.7,
            label="aggregate ECOD",
        )
        if len(smoothed_scores) == len(timestamps):
            score_axis.plot(
                timestamps,
                smoothed_scores,
                color="#111827",
                linewidth=1.4,
                label="smoothed aggregate",
            )
        for index, peak in enumerate(peaks):
            peak_sec = float(peak["peak_sec"])
            score_axis.axvline(
                peak_sec,
                color="#dc2626",
                linewidth=1.2,
                alpha=0.9,
                label="selected PRISM peak" if index == 0 else None,
            )
        score_axis.set_ylabel("ECOD score")
        score_axis.grid(True, alpha=0.2)
        score_axis.legend(loc="upper right")

        labels = list(annotation_labels)
        unknown_labels = sorted({event["label"] for event in events}.difference(labels))
        labels.extend(unknown_labels)
        label_positions = {label: index for index, label in enumerate(labels)}
        colors = plt.get_cmap("tab10")
        for event in events:
            label = event["label"]
            y_value = label_positions[label]
            start_sec = float(event["start_sec"])
            end_sec = float(event["end_sec"])
            reference_sec = float(event["reference_sec"])
            color = colors(y_value % 10)
            if end_sec > start_sec:
                annotation_axis.plot(
                    [start_sec, end_sec],
                    [y_value, y_value],
                    color=color,
                    linewidth=7,
                    solid_capstyle="butt",
                )
            annotation_axis.scatter([reference_sec], [y_value], color=color, s=24, zorder=3)
        annotation_axis.set_yticks(range(len(labels)), labels)
        annotation_axis.set_ylim(-0.75, max(0.75, len(labels) - 0.25))
        annotation_axis.set_xlabel("Time in trajectory (s)")
        annotation_axis.set_ylabel("Ground truth")
        annotation_axis.grid(True, axis="x", alpha=0.2)
        figure.suptitle(f"{bundle_id}: PRISM peaks and six-category ground truth")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=170)
        plt.close(figure)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip("._") or "unnamed"


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("available") is not True:
        raise ValueError(f"Aggregate ECOD debug data is unavailable: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
