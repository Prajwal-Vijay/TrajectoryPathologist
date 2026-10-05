"""Command-line orchestration for the ABC Qwen3-VL experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIRECTORY.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from VL_implemenation.abc_data import (
    Episode,
    discover_episodes,
    instruction_event_type,
)
from VL_implemenation.evaluation import (
    evaluate_localizations,
    write_evaluation_outputs,
)
from VL_implemenation.mosaic import (
    export_episode_mosaic,
    load_mosaic_timeline,
    mosaic_paths,
    video_sec_to_trajectory_sec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL K=1 insertion/removal localization on ABC-130K."
    )
    parser.add_argument(
        "command",
        choices=["inspect", "mosaic", "sanity", "infer", "evaluate", "full"],
    )
    parser.add_argument(
        "--config",
        default=str(PACKAGE_DIRECTORY / "config.json"),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--sanity-cases", type=int, default=10)
    parser.add_argument("--bundle-id", action="append", default=[])
    parser.add_argument("--overwrite-mosaics", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    episodes = _selected_episodes(
        discover_episodes(config["data_root"]),
        bundle_ids=set(args.bundle_id),
        max_episodes=args.max_episodes,
    )
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_dataset_manifest(output_root, episodes, config)

    if args.command == "inspect":
        print(json.dumps(_dataset_summary(episodes), indent=2, sort_keys=True))
        return
    if args.command == "mosaic":
        _export_mosaics(
            episodes,
            config,
            overwrite=args.overwrite_mosaics,
            verbose=args.verbose,
        )
        return
    if args.command == "evaluate":
        summary = _evaluate(output_root, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "sanity":
        summary = _run_sanity(
            episodes,
            config,
            case_count=args.sanity_cases,
            overwrite_mosaics=args.overwrite_mosaics,
            verbose=args.verbose,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "infer":
        _run_inference(
            episodes,
            config,
            build_mosaics=False,
            overwrite_mosaics=args.overwrite_mosaics,
            resume=not args.no_resume,
            verbose=args.verbose,
        )
    elif args.command == "full":
        _run_inference(
            episodes,
            config,
            build_mosaics=True,
            overwrite_mosaics=args.overwrite_mosaics,
            resume=not args.no_resume,
            verbose=args.verbose,
        )
    summary = _evaluate(output_root, config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _selected_episodes(
    episodes: list[Episode],
    *,
    bundle_ids: set[str],
    max_episodes: int | None,
) -> list[Episode]:
    if bundle_ids:
        episodes = [episode for episode in episodes if episode.bundle_id in bundle_ids]
        missing = bundle_ids - {episode.bundle_id for episode in episodes}
        if missing:
            raise ValueError(f"Unknown bundle IDs: {', '.join(sorted(missing))}")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be positive.")
        episodes = episodes[:max_episodes]
    return episodes


def _export_mosaics(
    episodes: list[Episode],
    config: dict[str, Any],
    *,
    overwrite: bool,
    verbose: bool,
) -> None:
    started = time.perf_counter()
    for index, episode in enumerate(episodes, start=1):
        metadata = _export_one_mosaic(episode, config, overwrite=overwrite)
        if verbose:
            print(
                f"[{index}/{len(episodes)}] {episode.task_name}/{episode.bundle_id} "
                f"{metadata['frame_count']} frames",
                flush=True,
            )
    print(
        f"Exported {len(episodes)} mosaics in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )


def _export_one_mosaic(
    episode: Episode,
    config: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    return export_episode_mosaic(
        episode,
        frame_cache_root=Path(config["frame_cache_root"]),
        mosaic_root=Path(config["mosaic_root"]),
        camera_sample_hz=float(config["camera_sample_hz"]),
        panel_height=int(config["panel_height"]),
        overwrite=overwrite,
    )


def _run_inference(
    episodes: list[Episode],
    config: dict[str, Any],
    *,
    build_mosaics: bool,
    overwrite_mosaics: bool,
    resume: bool,
    verbose: bool,
) -> None:
    from VL_implemenation.qwen_localizer import QwenLocalizer

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "localizations.jsonl"
    existing = _latest_rows(jsonl_path) if resume else {}
    if not resume:
        jsonl_path.write_text("", encoding="utf-8")

    localizer = QwenLocalizer(
        model_path=config["model_path"],
        model_sample_fps=float(config["model_sample_fps"]),
        min_pixels=int(config["min_pixels"]),
        max_pixels=int(config["max_pixels"]),
        total_pixels=int(config["total_pixels"]),
        scan_total_pixels=int(config["scan_total_pixels"]),
        max_new_tokens=int(config["max_new_tokens"]),
        seed=int(config["seed"]),
    )
    run_started = time.perf_counter()
    for index, episode in enumerate(episodes, start=1):
        previous = existing.get(episode.bundle_id)
        if resume and previous and previous.get("status") == "complete":
            if verbose:
                print(
                    f"[{index}/{len(episodes)}] resume skip {episode.bundle_id}",
                    flush=True,
                )
            continue
        episode_started = time.perf_counter()
        try:
            if build_mosaics:
                mosaic_metadata = _export_one_mosaic(
                    episode,
                    config,
                    overwrite=overwrite_mosaics,
                )
            else:
                paths = mosaic_paths(Path(config["mosaic_root"]), episode)
                if not paths["metadata"].is_file():
                    raise FileNotFoundError(
                        f"Missing mosaic metadata for {episode.bundle_id}; "
                        "run the mosaic or full command first."
                    )
                mosaic_metadata = json.loads(
                    paths["metadata"].read_text(encoding="utf-8")
                )
            timeline = load_mosaic_timeline(
                Path(mosaic_metadata["timeline_path"])
            )
            result = _windowed_localization(
                localizer=localizer,
                episode=episode,
                video_path=Path(mosaic_metadata["video_path"]),
                video_duration_sec=float(mosaic_metadata["video_duration_sec"]),
                window_sec=float(config["window_sec"]),
                stride_sec=float(config["window_stride_sec"]),
                fine_window_sec=float(config["fine_window_sec"]),
                explicit_task_yes_threshold=float(
                    config["explicit_task_yes_threshold"]
                ),
                generic_task_yes_threshold=float(
                    config["generic_task_yes_threshold"]
                ),
            )
            if isinstance(result.get("event"), dict):
                event = dict(result["event"])
                event["trajectory_start_sec"] = video_sec_to_trajectory_sec(
                    timeline, float(event["start_sec"])
                )
                event["trajectory_end_sec"] = video_sec_to_trajectory_sec(
                    timeline, float(event["end_sec"])
                )
                event["trajectory_midpoint_sec"] = video_sec_to_trajectory_sec(
                    timeline, float(event["midpoint_sec"])
                )
                result["event"] = event
            row = {
                "status": "complete",
                **episode.to_dict(),
                "mosaic": mosaic_metadata,
                **result,
                "episode_wall_sec": time.perf_counter() - episode_started,
            }
        except Exception as exc:
            row = {
                "status": "error",
                **episode.to_dict(),
                "parse_status": "runtime_error",
                "parse_error": f"{type(exc).__name__}: {exc}",
                "event": None,
                "episode_wall_sec": time.perf_counter() - episode_started,
            }
        _append_jsonl(jsonl_path, row)
        existing[episode.bundle_id] = row
        _write_run_summary(
            output_root,
            config,
            episodes,
            existing,
            status="running",
            elapsed_sec=time.perf_counter() - run_started,
        )
        if verbose or row["status"] == "error":
            print(
                f"[{index}/{len(episodes)}] {episode.task_name}/"
                f"{episode.bundle_id}: {row['status']} "
                f"{row.get('parse_status')}",
                flush=True,
            )
    _write_run_summary(
        output_root,
        config,
        episodes,
        existing,
        status="complete",
        elapsed_sec=time.perf_counter() - run_started,
    )


def _windowed_localization(
    *,
    localizer: Any,
    episode: Episode,
    video_path: Path,
    video_duration_sec: float,
    window_sec: float,
    stride_sec: float,
    fine_window_sec: float,
    explicit_task_yes_threshold: float,
    generic_task_yes_threshold: float,
) -> dict[str, Any]:
    explicit_type = instruction_event_type(episode.instruction)
    yes_probability_threshold = (
        explicit_task_yes_threshold
        if explicit_type is not None
        else generic_task_yes_threshold
    )
    windows = _temporal_windows(
        video_duration_sec,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    scores = []
    selected = None
    for window_start, window_end in windows:
        score = localizer.score_window(
            video_path=video_path,
            instruction=episode.instruction,
            window_start_sec=window_start,
            window_end_sec=window_end,
        )
        scores.append(
            {
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                **score,
            }
        )
        if float(score["yes_probability"]) > yes_probability_threshold:
            selected = scores[-1]
            break
    if selected is None:
        return {
            "raw_answer": None,
            "prompt": None,
            "parse_status": "valid_no_event",
            "parse_error": None,
            "parsed_payload": {"events": []},
            "event": None,
            "returned_event_count": 0,
            "window_scores": scores,
            "selected_window": None,
            "window_protocol": "chronological_yes_no_scan",
            "yes_probability_threshold": yes_probability_threshold,
            "total_model_sec": sum(
                float(score["total_model_sec"]) for score in scores
            ),
        }
    coarse_start = float(selected["window_start_sec"])
    coarse_end = float(selected["window_end_sec"])
    fine_windows = _temporal_windows(
        coarse_end - coarse_start,
        window_sec=fine_window_sec,
        stride_sec=fine_window_sec,
    )
    fine_scores = []
    for relative_start, relative_end in fine_windows:
        fine_start = coarse_start + relative_start
        fine_end = coarse_start + relative_end
        score = localizer.score_window(
            video_path=video_path,
            instruction=episode.instruction,
            window_start_sec=fine_start,
            window_end_sec=fine_end,
        )
        fine_scores.append(
            {
                "window_start_sec": fine_start,
                "window_end_sec": fine_end,
                **score,
            }
        )
    fine_selected = max(
        fine_scores,
        key=lambda score: float(score["yes_probability"]),
    )
    window_start = float(fine_selected["window_start_sec"])
    window_end = float(fine_selected["window_end_sec"])
    localized = localizer.localize_window(
        video_path=video_path,
        instruction=episode.instruction,
        window_start_sec=window_start,
        window_end_sec=window_end,
    )
    if isinstance(localized.get("event"), dict):
        event = dict(localized["event"])
        event["window_local_start_sec"] = event["start_sec"]
        event["window_local_end_sec"] = event["end_sec"]
        event["window_local_midpoint_sec"] = event["midpoint_sec"]
        event["start_sec"] = window_start + float(event["start_sec"])
        event["end_sec"] = window_start + float(event["end_sec"])
        event["midpoint_sec"] = window_start + float(event["midpoint_sec"])
        localized["event"] = event
        if explicit_type is not None and event["type"] != explicit_type:
            event["model_reported_type"] = event["type"]
            event["type"] = explicit_type
    localized["window_scores"] = scores
    localized["fine_window_scores"] = fine_scores
    localized["selected_window"] = {
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "yes_probability": fine_selected["yes_probability"],
        "no_probability": fine_selected["no_probability"],
        "coarse_window_start_sec": coarse_start,
        "coarse_window_end_sec": coarse_end,
        "coarse_yes_probability": selected["yes_probability"],
    }
    localized["window_protocol"] = "coarse_to_fine_yes_no_scan"
    localized["yes_probability_threshold"] = yes_probability_threshold
    localized["total_model_sec"] = float(localized["total_model_sec"]) + sum(
        float(score["total_model_sec"]) for score in scores
    ) + sum(
        float(score["total_model_sec"]) for score in fine_scores
    )
    return localized


def _temporal_windows(
    duration_sec: float,
    *,
    window_sec: float,
    stride_sec: float,
) -> list[tuple[float, float]]:
    if duration_sec <= 0 or window_sec <= 0 or stride_sec <= 0:
        raise ValueError("Duration, window, and stride must be positive.")
    output = []
    start_sec = 0.0
    while start_sec < duration_sec:
        end_sec = min(duration_sec, start_sec + window_sec)
        output.append((start_sec, end_sec))
        if end_sec >= duration_sec:
            break
        start_sec += stride_sec
    return output


def _run_sanity(
    episodes: list[Episode],
    config: dict[str, Any],
    *,
    case_count: int,
    overwrite_mosaics: bool,
    verbose: bool,
) -> dict[str, Any]:
    from statistics import median

    from VL_implemenation.qwen_localizer import QwenLocalizer
    from VL_implemenation.sanity import measured_gripper_closures

    if case_count <= 0:
        raise ValueError("--sanity-cases must be positive.")
    candidates: list[tuple[Episode, list[dict[str, Any]]]] = []
    for episode in sorted(
        episodes,
        key=lambda item: (not bool(item.ground_intervals), item.task_name, item.bundle_id),
    ):
        transitions = measured_gripper_closures(episode)
        if transitions:
            candidates.append((episode, transitions))
        if len(candidates) == case_count:
            break
    if len(candidates) < case_count:
        raise ValueError(
            f"Found only {len(candidates)}/{case_count} clear gripper sanity cases."
        )

    localizer = QwenLocalizer(
        model_path=config["model_path"],
        model_sample_fps=float(config["model_sample_fps"]),
        min_pixels=int(config["min_pixels"]),
        max_pixels=int(config["max_pixels"]),
        total_pixels=int(config["total_pixels"]),
        scan_total_pixels=int(config["scan_total_pixels"]),
        max_new_tokens=int(config["max_new_tokens"]),
        seed=int(config["seed"]),
    )
    rows: list[dict[str, Any]] = []
    for index, (episode, transitions) in enumerate(candidates, start=1):
        mosaic_metadata = _export_one_mosaic(
            episode,
            config,
            overwrite=overwrite_mosaics,
        )
        timeline = load_mosaic_timeline(Path(mosaic_metadata["timeline_path"]))
        result = localizer.localize_gripper_close(
            video_path=Path(mosaic_metadata["video_path"]),
            video_duration_sec=float(mosaic_metadata["video_duration_sec"]),
        )
        predicted_trajectory_sec = None
        nearest_error_sec = None
        first_error_sec = None
        if isinstance(result.get("event"), dict):
            predicted_trajectory_sec = video_sec_to_trajectory_sec(
                timeline,
                float(result["event"]["midpoint_sec"]),
            )
            transition_times = [
                float(transition["timestamp_sec"]) for transition in transitions
            ]
            nearest_error_sec = min(
                abs(predicted_trajectory_sec - timestamp)
                for timestamp in transition_times
            )
            first_error_sec = abs(predicted_trajectory_sec - transition_times[0])
        row = {
            "task_name": episode.task_name,
            "bundle_id": episode.bundle_id,
            "measured_transitions": transitions,
            "predicted_trajectory_sec": predicted_trajectory_sec,
            "nearest_transition_abs_error_sec": nearest_error_sec,
            "first_transition_abs_error_sec": first_error_sec,
            "mosaic": mosaic_metadata,
            **result,
        }
        rows.append(row)
        if verbose:
            print(
                f"[sanity {index}/{case_count}] {episode.bundle_id}: "
                f"nearest error={nearest_error_sec}",
                flush=True,
            )
    valid_errors = [
        float(row["nearest_transition_abs_error_sec"])
        for row in rows
        if row["nearest_transition_abs_error_sec"] is not None
    ]
    summary = {
        "case_count": len(rows),
        "valid_prediction_count": len(valid_errors),
        "median_nearest_transition_abs_error_sec": (
            median(valid_errors) if valid_errors else None
        ),
        "max_nearest_transition_abs_error_sec": (
            max(valid_errors) if valid_errors else None
        ),
        "note": (
            "Measured state-threshold transitions are an alignment diagnostic; "
            "they are not guaranteed object-contact annotations."
        ),
    }
    output_root = Path(config["output_root"])
    _write_jsonl_atomic(output_root / "gripper_sanity.jsonl", rows)
    _write_json(output_root / "gripper_sanity_summary.json", summary)
    return summary


def _evaluate(output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = list(_latest_rows(output_root / "localizations.jsonl").values())
    if not rows:
        raise ValueError("No localization rows are available to evaluate.")
    summary, scored = evaluate_localizations(
        rows,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    summary["protocol"] = {
        "k": 1,
        "prediction": "first chronological returned interval",
        "window_protocol": "coarse chronological scan, fine max-probability scan",
        "window_sec": config["window_sec"],
        "window_stride_sec": config["window_stride_sec"],
        "fine_window_sec": config["fine_window_sec"],
        "scan_total_pixels": config["scan_total_pixels"],
        "localization_total_pixels": config["total_pixels"],
        "explicit_task_yes_threshold": config["explicit_task_yes_threshold"],
        "generic_task_yes_threshold": config["generic_task_yes_threshold"],
        "explicit_task_type_policy": (
            "Use the task instruction's insert/remove verb when explicit; "
            "retain the model-reported type in model_reported_type."
        ),
        "point": "clipped interval midpoint mapped through mosaic timeline",
        "primary_hit": "same-type midpoint in half-open ground interval",
        "unparseable_policy": "miss",
        "ground_aliases": {
            "insertion": ["Insert ...", "Press the plug into ..."],
            "removal": ["Remove ...", "Take out ..."],
        },
    }
    write_evaluation_outputs(output_root, summary, scored)
    return summary


def _dataset_summary(episodes: list[Episode]) -> dict[str, Any]:
    return {
        "episode_count": len(episodes),
        "event_episode_count": sum(bool(ep.ground_intervals) for ep in episodes),
        "no_event_episode_count": sum(not ep.ground_intervals for ep in episodes),
        "ground_interval_count": sum(len(ep.ground_intervals) for ep in episodes),
        "insertion_interval_count": sum(
            interval.event_type == "insertion"
            for ep in episodes
            for interval in ep.ground_intervals
        ),
        "removal_interval_count": sum(
            interval.event_type == "removal"
            for ep in episodes
            for interval in ep.ground_intervals
        ),
    }


def _write_dataset_manifest(
    output_root: Path,
    episodes: list[Episode],
    config: dict[str, Any],
) -> None:
    payload = {
        "summary": _dataset_summary(episodes),
        "data_root": config["data_root"],
        "episodes": [episode.to_dict() for episode in episodes],
    }
    _write_json(output_root / "dataset_manifest.json", payload)


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return output
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output[str(row["bundle_id"])] = row
    return output


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_run_summary(
    output_root: Path,
    config: dict[str, Any],
    episodes: list[Episode],
    rows: dict[str, dict[str, Any]],
    *,
    status: str,
    elapsed_sec: float,
) -> None:
    selected_ids = {episode.bundle_id for episode in episodes}
    selected_rows = [row for key, row in rows.items() if key in selected_ids]
    _write_json(
        output_root / "run_summary.json",
        {
            "status": status,
            "requested_episode_count": len(episodes),
            "completed_episode_count": sum(
                row.get("status") == "complete" for row in selected_rows
            ),
            "error_episode_count": sum(
                row.get("status") == "error" for row in selected_rows
            ),
            "elapsed_sec": elapsed_sec,
            "config": config,
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
