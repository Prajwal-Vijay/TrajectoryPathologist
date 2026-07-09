"""End-to-end runner for metric-driven key event discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pathologist.adapters.roboeval_bundle import RoboEvalBundleAdapter
from pathologist.adapters.vifailback_bundle import ViFailBackBundleAdapter
from pathologist.config import config_section, load_pipeline_config
from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.schemas.metrics import MetricProfile, MetricResult
from pathologist.schemas import KeyEvent, TaskContract
from pathologist.vlm import OpenAIEventLabelClient, VLMEventLabeler


def run_key_event_pipeline(
    bundle_dir: str | Path,
    output_dir: str | Path | None = None,
    task_contract: TaskContract | None = None,
    task_contract_path: str | Path | None = None,
    config_path: str | Path | None = None,
    adapter: str = "roboeval",
    video_path: str | Path | None = None,
    frame_cache_dir: str | Path | None = None,
    video_sample_fps: float = 2.0,
    hdf5_fps: float = 25.0,
    default_control_hz: float = 50.0,
    frame_stride_within_window: int | None = None,
    max_events_to_label: int | None = None,
    openai_model: str | None = None,
    event_label_client: Any | None = None,
    skip_labeling: bool = False,
    plot_left_tail: bool = False,
    plot_right_tail: bool = False,
    plot_skew_tail: bool = False,
    smooth_window: int = 1,
) -> dict[str, Any]:
    """Run the current key-event pipeline on one trajectory folder."""

    bundle_path = Path(bundle_dir).expanduser().resolve()
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _default_output_path(bundle_path)
    )
    output_path.mkdir(parents=True, exist_ok=True)

    config_bundle_path = bundle_path if bundle_path.is_dir() else bundle_path.parent
    resolved_task_contract = task_contract or _load_task_contract(task_contract_path, config_bundle_path)
    config = load_pipeline_config(config_path=config_path, bundle_path=config_bundle_path)
    selector_config = config_section(config, "selector")
    labeling_config = config_section(config, "labeling")

    resolved_frame_stride_within_window = (
        int(frame_stride_within_window)
        if frame_stride_within_window is not None
        else int(selector_config.get("frame_stride_within_window", 10))
    )
    metric_packet_context_steps = int(selector_config.get("metric_packet_context_steps", 10))
    visual_context_sec = float(selector_config.get("visual_context_sec", 1.0))
    max_frames_per_camera = int(selector_config.get("max_frames_per_camera", 5))
    surprise_flood_config = selector_config.get("surprise_flood", {})
    if not isinstance(surprise_flood_config, dict):
        surprise_flood_config = {}

    resolved_max_events_to_label = (
        int(max_events_to_label)
        if max_events_to_label is not None
        else int(labeling_config.get("max_events_to_label", 8))
    )
    resolved_openai_model = openai_model or labeling_config.get("openai_model")

    bundle = _load_bundle(
        adapter=adapter,
        bundle_path=bundle_path,
        video_path=video_path,
        task_contract=resolved_task_contract,
        frame_cache_dir=frame_cache_dir or output_path / "frame_cache",
        video_sample_fps=video_sample_fps,
        hdf5_fps=hdf5_fps,
        default_control_hz=default_control_hz,
    )

    metric_profile = MetricEngine(config=config).compute(bundle)
    ecod_plot_summary = _write_ecod_score_plots(
        metric_profile,
        config,
        output_path,
        plot_left_tail=plot_left_tail,
        plot_right_tail=plot_right_tail,
        plot_skew_tail=plot_skew_tail,
        smooth_window=smooth_window,
    )
    metric_profile_path = output_path / "metric_profile.json"
    metric_profile_path.write_text(json.dumps(metric_profile.to_dict(), indent=2), encoding="utf-8")

    selector = MetricEventSelector(
        frame_stride_within_window=resolved_frame_stride_within_window,
        visual_context_sec=visual_context_sec,
        max_frames_per_camera=max_frames_per_camera,
        metric_packet_context_steps=metric_packet_context_steps,
        surprise_flood_config=surprise_flood_config,
    )
    debug_timeline_path = output_path / "timeline_debug.json"
    proposals = selector.select(
        metric_profile,
        trajectory_bundle=bundle,
        debug_timeline_path=debug_timeline_path,
    )
    proposals_path = output_path / "key_event_proposals.json"
    proposals_path.write_text(
        json.dumps([proposal.to_dict() for proposal in proposals], indent=2),
        encoding="utf-8",
    )

    labeled_key_events: list[KeyEvent] = []
    if not skip_labeling and resolved_max_events_to_label > 0:
        client = event_label_client or OpenAIEventLabelClient(model=resolved_openai_model)
        labeled_key_events = VLMEventLabeler(client).label_many(
            proposals[:resolved_max_events_to_label],
            resolved_task_contract,
        )
    labeled_key_events_path = output_path / "key_events.json"
    labeled_key_events_path.write_text(
        json.dumps([key_event.to_dict() for key_event in labeled_key_events], indent=2),
        encoding="utf-8",
    )
    visual_event_labels_path = output_path / "visual_event_labels.json"
    visual_event_labels_path.write_text(
        json.dumps(
            [
                {
                    "event_id": key_event.proposal.event_id,
                    "visual_label": key_event.label.metadata.get("visual_label") if key_event.label else None,
                    "raw_visual_response": key_event.label.metadata.get("raw_visual_response") if key_event.label else None,
                }
                for key_event in labeled_key_events
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    metric_grounding_cards_path = output_path / "metric_grounding_cards.json"
    metric_grounding_cards_path.write_text(
        json.dumps(
            [
                {
                    "event_id": key_event.proposal.event_id,
                    "metric_grounding_cards": (
                        key_event.label.metadata.get("metric_grounding_cards")
                        if key_event.label
                        else []
                    ),
                }
                for key_event in labeled_key_events
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    task_contract_path_out = output_path / "task_contract.json"
    task_contract_path_out.write_text(
        json.dumps(resolved_task_contract.to_dict(), indent=2),
        encoding="utf-8",
    )
    effective_config_path = output_path / "pathologist_config_effective.json"
    effective_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_summary = {
        "bundle_dir": str(bundle_path),
        "adapter": adapter,
        "video_path": str(Path(video_path).expanduser().resolve()) if video_path is not None else None,
        "output_dir": str(output_path),
        "bundle_id": bundle.bundle_id,
        "proposal_count": len(proposals),
        "labeled_key_event_count": len(labeled_key_events),
        "metric_profile_path": str(metric_profile_path),
        "timeline_debug_path": str(debug_timeline_path),
        "proposals_path": str(proposals_path),
        "key_events_path": str(labeled_key_events_path),
        "visual_event_labels_path": str(visual_event_labels_path),
        "metric_grounding_cards_path": str(metric_grounding_cards_path),
        "task_contract_path": str(task_contract_path_out),
        "effective_config_path": str(effective_config_path),
        "ecod_score_plots": ecod_plot_summary,
    }
    summary_path = output_path / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def _write_ecod_score_plots(
    metric_profile: MetricProfile,
    config: dict[str, Any],
    output_path: Path,
    plot_left_tail: bool = False,
    plot_right_tail: bool = False,
    plot_skew_tail: bool = False,
    smooth_window: int = 1,
) -> dict[str, Any]:
    surprise_config = config_section(config, "surprise")
    plot_config = surprise_config.get("plot", {})
    if not isinstance(plot_config, dict):
        plot_config = {}
    resolved_smooth_window = max(1, int(smooth_window))
    if not bool(surprise_config.get("enabled", True)) or not bool(plot_config.get("enabled", True)):
        return {"enabled": False, "smooth_window": resolved_smooth_window, "plots": []}

    output_dir = output_path / str(plot_config.get("output_dir_name", "ecod_score_plots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "plot_status.json"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        status = {
            "enabled": True,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "smooth_window": resolved_smooth_window,
            "plots": [],
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    metric_lookup = {metric.name: metric for metric in metric_profile.metrics}
    plot_paths: list[str] = []
    selected_components = _selected_ecod_plot_components(
        plot_left_tail=plot_left_tail,
        plot_right_tail=plot_right_tail,
        plot_skew_tail=plot_skew_tail,
    )
    aggregate_metric_name = str(surprise_config.get("aggregate_metric_name", "ecod_aggregate_score"))
    aggregate_metric = metric_lookup.get(aggregate_metric_name)
    if aggregate_metric is not None and aggregate_metric.samples:
        aggregate_plot_path = output_dir / "ecod_aggregate_score.png"
        try:
            _plot_ecod_aggregate_metric(aggregate_metric, aggregate_plot_path, plt, selected_components)
            plot_paths.append(str(aggregate_plot_path))
        except Exception as exc:
            plot_paths.append(f"failed:{aggregate_metric.name}:{type(exc).__name__}:{exc}")
        if resolved_smooth_window > 1:
            smoothed_plot_path = output_dir / f"ecod_aggregate_score_smoothed_w{resolved_smooth_window}.png"
            try:
                _plot_ecod_aggregate_smoothed_metric(
                    aggregate_metric,
                    smoothed_plot_path,
                    plt,
                    window=resolved_smooth_window,
                )
                plot_paths.append(str(smoothed_plot_path))
            except Exception as exc:
                plot_paths.append(f"failed:{aggregate_metric.name}_smoothed:{type(exc).__name__}:{exc}")

    for score_metric in metric_profile.metrics:
        if not score_metric.name.endswith("_ecod_score"):
            continue
        source_metric_name = score_metric.metadata.get("source_metric_name")
        if not isinstance(source_metric_name, str):
            continue
        source_metric = metric_lookup.get(source_metric_name)
        if source_metric is None or not source_metric.samples or not score_metric.samples:
            continue
        plot_path = output_dir / f"{score_metric.name}.png"
        try:
            _plot_ecod_score_metric(source_metric, score_metric, plot_path, plt, selected_components)
            plot_paths.append(str(plot_path))
        except Exception as exc:
            plot_paths.append(f"failed:{score_metric.name}:{type(exc).__name__}:{exc}")

    status = {
        "enabled": True,
        "available": True,
        "output_dir": str(output_dir),
        "selected_components": sorted(selected_components),
        "smooth_window": resolved_smooth_window,
        "plots": plot_paths,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _plot_ecod_score_metric(
    source_metric: MetricResult,
    score_metric: MetricResult,
    plot_path: Path,
    plt: Any,
    selected_components: set[str],
) -> None:
    source_times = [sample.timestamp_sec for sample in source_metric.samples if isinstance(sample.value, (int, float, bool))]
    source_values = [float(sample.value) for sample in source_metric.samples if isinstance(sample.value, (int, float, bool))]
    score_times = [sample.timestamp_sec for sample in score_metric.samples]
    score_values = [float(sample.value) for sample in score_metric.samples]
    if not source_times or not score_times:
        return

    figure, raw_axis = plt.subplots(figsize=(10, 4))
    score_axis = raw_axis.twinx()
    left_tail_scores = _float_metadata_list(score_metric, "left_tail_scores")
    right_tail_scores = _float_metadata_list(score_metric, "right_tail_scores")
    skew_tail_scores = _float_metadata_list(score_metric, "skew_tail_scores")
    raw_axis.scatter(source_times, source_values, color="#1f77b4", s=8, alpha=0.65, label=source_metric.name)
    if "actual" in selected_components:
        score_axis.scatter(score_times, score_values, color="#111111", s=10, alpha=0.75, label=score_metric.name)
    if "left" in selected_components and len(left_tail_scores) == len(score_times):
        score_axis.scatter(score_times, left_tail_scores, color="#2ca02c", s=7, alpha=0.55, label="left_tail_score")
    if "right" in selected_components and len(right_tail_scores) == len(score_times):
        score_axis.scatter(score_times, right_tail_scores, color="#d62728", s=7, alpha=0.55, label="right_tail_score")
    if "skew" in selected_components and len(skew_tail_scores) == len(score_times):
        score_axis.scatter(score_times, skew_tail_scores, color="#9467bd", s=7, alpha=0.55, label="skew_tail_score")
    raw_axis.set_xlabel("timestamp_sec")
    raw_axis.set_ylabel(source_metric.unit or "raw value", color="#1f77b4")
    score_axis.set_ylabel("ECOD surprise score", color="#d62728")
    raw_axis.tick_params(axis="y", labelcolor="#1f77b4")
    score_axis.tick_params(axis="y", labelcolor="#d62728")
    raw_axis.grid(True, alpha=0.25)
    score_axis.legend(loc="best")
    figure.suptitle(f"{source_metric.name} raw value and ECOD surprise score")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _plot_ecod_aggregate_metric(
    aggregate_metric: MetricResult,
    plot_path: Path,
    plt: Any,
    selected_components: set[str],
) -> None:
    timestamps = [sample.timestamp_sec for sample in aggregate_metric.samples]
    values = [float(sample.value) for sample in aggregate_metric.samples]
    if not timestamps:
        return
    left_tail_scores = _float_metadata_list(aggregate_metric, "aggregate_left_tail_scores")
    right_tail_scores = _float_metadata_list(aggregate_metric, "aggregate_right_tail_scores")
    skew_tail_scores = _float_metadata_list(aggregate_metric, "aggregate_skew_tail_scores")

    figure, axis = plt.subplots(figsize=(10, 4))
    if "actual" in selected_components:
        axis.scatter(timestamps, values, color="#111111", s=11, alpha=0.8, label=aggregate_metric.name)
    if "left" in selected_components and len(left_tail_scores) == len(timestamps):
        axis.scatter(timestamps, left_tail_scores, color="#2ca02c", s=8, alpha=0.55, label="aggregate_left_tail_score")
    if "right" in selected_components and len(right_tail_scores) == len(timestamps):
        axis.scatter(timestamps, right_tail_scores, color="#d62728", s=8, alpha=0.55, label="aggregate_right_tail_score")
    if "skew" in selected_components and len(skew_tail_scores) == len(timestamps):
        axis.scatter(timestamps, skew_tail_scores, color="#9467bd", s=8, alpha=0.45, label="aggregate_skew_tail_score")
    axis.set_xlabel("timestamp_sec")
    axis.set_ylabel("Aggregate ECOD surprise score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.suptitle("ECOD aggregate surprise score across metrics")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _plot_ecod_aggregate_smoothed_metric(
    aggregate_metric: MetricResult,
    plot_path: Path,
    plt: Any,
    window: int,
) -> None:
    timestamps = [sample.timestamp_sec for sample in aggregate_metric.samples]
    values = [float(sample.value) for sample in aggregate_metric.samples]
    if not timestamps:
        return
    smoothed_values = _centered_moving_average(values, window)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(timestamps, values, color="#999999", s=6, alpha=0.25, label="raw_aggregate_score")
    axis.scatter(timestamps, smoothed_values, color="#111111", s=11, alpha=0.8, label=f"smoothed_w{window}")
    axis.set_xlabel("timestamp_sec")
    axis.set_ylabel("Aggregate ECOD surprise score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.suptitle(f"Smoothed ECOD aggregate surprise score (window={window} samples)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _centered_moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return list(values)
    radius_left = (window - 1) // 2
    radius_right = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius_left)
        end = min(len(values), index + radius_right + 1)
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def _selected_ecod_plot_components(
    plot_left_tail: bool,
    plot_right_tail: bool,
    plot_skew_tail: bool,
) -> set[str]:
    selected = set()
    if plot_left_tail:
        selected.add("left")
    if plot_right_tail:
        selected.add("right")
    if plot_skew_tail:
        selected.add("skew")
    return selected or {"actual"}


def _float_metadata_list(metric: MetricResult, key: str) -> list[float]:
    values = metric.metadata.get(key)
    if not isinstance(values, list):
        return []
    output = []
    for value in values:
        if isinstance(value, (int, float, bool)):
            output.append(float(value))
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the metric-first key event pipeline.")
    parser.add_argument(
        "bundle_dir",
        help="Path to the RoboEval-style trajectory folder, or a ViFailBack .h5 episode when --adapter vifailback is used.",
    )
    parser.add_argument(
        "--adapter",
        choices=["roboeval", "vifailback"],
        default="roboeval",
        help="Input adapter to use.",
    )
    parser.add_argument(
        "--video-path",
        help="Optional fallback video/image/frame directory for --adapter vifailback when HDF5 RGB frames are unavailable.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        help="Directory for materialized ViFailBack HDF5/video frames. Defaults to output_dir/frame_cache.",
    )
    parser.add_argument(
        "--video-sample-fps",
        type=float,
        default=2.0,
        help="Frame extraction rate for ViFailBack videos.",
    )
    parser.add_argument(
        "--hdf5-fps",
        type=float,
        default=25.0,
        help="FPS used to timestamp ViFailBack HDF5 RGB/telemetry samples. Dataset visualization default is 25.",
    )
    parser.add_argument(
        "--default-control-hz",
        type=float,
        default=50.0,
        help="Fallback control rate for ViFailBack HDF5 telemetry if no video frames are available.",
    )
    parser.add_argument(
        "--task-contract",
        dest="task_contract_path",
        help="Optional path to a task_contract.json file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where pipeline outputs will be written.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Optional pathologist config JSON. Bundle-local pathologist_config.json is loaded automatically.",
    )
    parser.add_argument(
        "--frame-stride-within-window",
        type=int,
        default=None,
        help="Sample every Nth frame per camera inside each proposal window.",
    )
    parser.add_argument(
        "--max-events-to-label",
        type=int,
        default=None,
        help="Maximum number of top-ranked proposals to pass to the VLM.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="Optional model override for OpenAI event labeling.",
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Only compute metrics and key-event proposals, without VLM labels.",
    )
    parser.add_argument(
        "--left",
        dest="plot_left_tail",
        action="store_true",
        help="Plot ECOD left-tail score components instead of the default actual score.",
    )
    parser.add_argument(
        "--right",
        dest="plot_right_tail",
        action="store_true",
        help="Plot ECOD right-tail score components instead of the default actual score.",
    )
    parser.add_argument(
        "--skew",
        dest="plot_skew_tail",
        action="store_true",
        help="Plot ECOD skew-selected score components instead of the default actual score.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window, in samples, for an extra smoothed aggregate ECOD plot. Use 1 to disable.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_key_event_pipeline(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        task_contract_path=args.task_contract_path,
        config_path=args.config_path,
        adapter=args.adapter,
        video_path=args.video_path,
        frame_cache_dir=args.frame_cache_dir,
        video_sample_fps=args.video_sample_fps,
        hdf5_fps=args.hdf5_fps,
        default_control_hz=args.default_control_hz,
        frame_stride_within_window=args.frame_stride_within_window,
        max_events_to_label=args.max_events_to_label,
        openai_model=args.openai_model,
        skip_labeling=args.skip_labeling,
        plot_left_tail=args.plot_left_tail,
        plot_right_tail=args.plot_right_tail,
        plot_skew_tail=args.plot_skew_tail,
        smooth_window=args.smooth_window,
    )
    print(json.dumps(summary, indent=2))


def _load_bundle(
    adapter: str,
    bundle_path: Path,
    video_path: str | Path | None,
    task_contract: TaskContract,
    frame_cache_dir: str | Path | None,
    video_sample_fps: float,
    hdf5_fps: float,
    default_control_hz: float,
):
    if adapter == "roboeval":
        return RoboEvalBundleAdapter().load(bundle_path, task_contract=task_contract)
    if adapter == "vifailback":
        return ViFailBackBundleAdapter().load(
            h5_path=bundle_path,
            video_path=video_path,
            task_contract=task_contract,
            frame_cache_dir=frame_cache_dir,
            video_sample_fps=video_sample_fps,
            hdf5_fps=hdf5_fps,
            default_control_hz=default_control_hz,
        )
    raise ValueError(f"Unknown adapter: {adapter}")


def _default_output_path(bundle_path: Path) -> Path:
    if bundle_path.is_dir():
        return bundle_path / "pathologist_outputs"
    return bundle_path.parent / f"{bundle_path.stem}_pathologist_outputs"


def _load_task_contract(
    task_contract_path: str | Path | None,
    bundle_path: Path,
) -> TaskContract:
    if task_contract_path is not None:
        return _task_contract_from_json(Path(task_contract_path).expanduser().resolve())

    for candidate in (bundle_path / "task_contract.json", bundle_path.parent / "task_contract.json"):
        if candidate.exists():
            return _task_contract_from_json(candidate)

    task_name = bundle_path.name.replace("_", " ").strip().title() or "Robot Manipulation Task"
    return TaskContract(
        task_name=task_name,
        task_description=f"Robot manipulation trajectory for {task_name}.",
        success_criteria="Complete the intended manipulation successfully and stably.",
        failure_criteria="Fail, regress, or show suboptimal behavior during the intended manipulation.",
        extra_context={
            "task_contract_source": "synthetic_default",
            "outcome_reasoning_allowed": False,
        },
    )


def _task_contract_from_json(path: Path) -> TaskContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task contract JSON must contain an object: {path}")
    return TaskContract(**payload)


if __name__ == "__main__":
    main()
