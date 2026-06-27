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
    point_event_delta_sec: float | None = None,
    frame_stride_within_window: int | None = None,
    max_events_to_label: int | None = None,
    openai_model: str | None = None,
    event_label_client: Any | None = None,
    skip_labeling: bool = False,
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

    resolved_point_event_delta_sec = (
        float(point_event_delta_sec)
        if point_event_delta_sec is not None
        else float(selector_config.get("point_event_delta_sec", 0.2))
    )
    resolved_frame_stride_within_window = (
        int(frame_stride_within_window)
        if frame_stride_within_window is not None
        else int(selector_config.get("frame_stride_within_window", 10))
    )
    metric_packet_context_steps = int(selector_config.get("metric_packet_context_steps", 10))
    visual_context_sec = float(selector_config.get("visual_context_sec", 1.0))
    max_frames_per_camera = int(selector_config.get("max_frames_per_camera", 5))
    min_segment_duration_sec = float(selector_config.get("min_segment_duration_sec", 0.2))
    same_metric_merge_gap_sec = float(selector_config.get("same_metric_merge_gap_sec", 0.25))
    proposal_merge_gap_sec = float(selector_config.get("proposal_merge_gap_sec", 0.25))
    ignored_metric_names = selector_config.get("ignored_metric_names", ["timestamp_regularity"])
    if not isinstance(ignored_metric_names, list):
        ignored_metric_names = ["timestamp_regularity"]
    same_metric_merge_gap_excluded_metric_names = selector_config.get(
        "same_metric_merge_gap_excluded_metric_names",
        ["gripper_transition"],
    )
    if not isinstance(same_metric_merge_gap_excluded_metric_names, list):
        same_metric_merge_gap_excluded_metric_names = ["gripper_transition"]

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
    metric_profile_path = output_path / "metric_profile.json"
    metric_profile_path.write_text(json.dumps(metric_profile.to_dict(), indent=2), encoding="utf-8")

    selector = MetricEventSelector(
        point_event_delta_sec=resolved_point_event_delta_sec,
        ignored_metric_names={str(name) for name in ignored_metric_names},
        frame_stride_within_window=resolved_frame_stride_within_window,
        visual_context_sec=visual_context_sec,
        max_frames_per_camera=max_frames_per_camera,
        metric_packet_context_steps=metric_packet_context_steps,
        min_segment_duration_sec=min_segment_duration_sec,
        same_metric_merge_gap_sec=same_metric_merge_gap_sec,
        same_metric_merge_gap_excluded_metric_names={
            str(name)
            for name in same_metric_merge_gap_excluded_metric_names
        },
        proposal_merge_gap_sec=proposal_merge_gap_sec,
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
    }
    summary_path = output_path / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


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
        "--point-event-delta-sec",
        type=float,
        default=None,
        help="Half-window expansion for point metric events.",
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
        point_event_delta_sec=args.point_event_delta_sec,
        frame_stride_within_window=args.frame_stride_within_window,
        max_events_to_label=args.max_events_to_label,
        openai_model=args.openai_model,
        skip_labeling=args.skip_labeling,
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
