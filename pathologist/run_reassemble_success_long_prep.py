"""Prepare long successful REASSEMBLE high-level demos with trimmed videos."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from pathologist.adapters.reassemble_bundle import _global_timestamp_origin, _parse_segments
from pathologist.reassemble import run_reassemble_eval_pipeline


@dataclass(frozen=True, slots=True)
class SuccessCandidate:
    bundle_id: str
    h5_path: str
    source_file_bundle_id: str
    segment_id: str
    segment_index: int | None
    task_text: str
    task_kind: str
    start_sec_in_file: float
    end_sec_in_file: float
    duration_sec: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare long successful REASSEMBLE high-level demos for annotation."
    )
    parser.add_argument("--data-root", required=True, help="Directory containing REASSEMBLE .h5 files.")
    parser.add_argument(
        "--output-dir",
        default="REASSEMBLE_Successful_Long",
        help="Directory for long successful-demo outputs.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of selected peaks per selector.")
    parser.add_argument("--max-demos", type=int, default=50, help="Number of long demos to prepare.")
    parser.add_argument(
        "--balanced-count-per-kind",
        type=int,
        default=None,
        help="Select this many demos per requested task kind instead of taking --max-demos overall.",
    )
    parser.add_argument(
        "--task-kinds",
        nargs="+",
        default=["pick", "place", "insert", "remove"],
        help="Task kinds eligible for balanced selection.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=30.0,
        help="Minimum high-level action duration.",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=60.0,
        help="Maximum high-level action duration.",
    )
    parser.add_argument(
        "--selection",
        choices=["longest", "chronological"],
        default="longest",
        help="How to choose demos when more than --max-demos are eligible.",
    )
    parser.add_argument(
        "--include-no-action",
        action="store_true",
        help="Include successful 'No action.' segments if they meet the duration bounds.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Selectors to compare.",
    )
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default="persistence",
        help="Metric proposal ranking used by the viewer output.",
    )
    parser.add_argument("--config", dest="config_path", help="Optional pathologist config JSON override.")
    parser.add_argument(
        "--robot-downsample-hz",
        type=float,
        default=None,
        help="Downsample robot streams before metric computation.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default=None,
        help="Optional shared decoded-frame cache. Defaults to <output-dir>/frame_cache.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=["hand", "hama1", "hama2"],
        help="Cameras to decode and export.",
    )
    parser.add_argument(
        "--skip-trimmed-videos",
        action="store_true",
        help="Prepare viewer/eval outputs but do not create MP4 clips.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs instead of restarting eval rows.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = scan_success_candidates(
        data_root=data_root,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        include_no_action=args.include_no_action,
        task_kinds={str(kind).lower() for kind in args.task_kinds},
    )
    if args.balanced_count_per_kind is not None:
        selected = select_balanced_candidates(
            candidates,
            task_kinds=[str(kind).lower() for kind in args.task_kinds],
            count_per_kind=args.balanced_count_per_kind,
            selection=args.selection,
        )
        requested_count = args.balanced_count_per_kind * len(args.task_kinds)
    else:
        selected = select_candidates(candidates, max_demos=args.max_demos, selection=args.selection)
        requested_count = args.max_demos
    if len(selected) < requested_count:
        print(
            f"Only found {len(selected)} eligible demos in "
            f"[{args.min_duration_sec}, {args.max_duration_sec}] seconds."
        )

    _write_candidate_outputs(output_dir, candidates, selected)
    selected_bundle_ids = {candidate.bundle_id for candidate in selected}
    windows_by_h5 = defaultdict(list)
    candidates_by_h5: dict[str, list[SuccessCandidate]] = defaultdict(list)
    for candidate in selected:
        windows_by_h5[candidate.h5_path].append(
            (candidate.start_sec_in_file, candidate.end_sec_in_file)
        )
        candidates_by_h5[candidate.h5_path].append(candidate)

    summary: dict[str, Any] | None = None
    for h5_path in sorted(candidates_by_h5, key=lambda path: min(c.bundle_id for c in candidates_by_h5[path])):
        if args.verbose:
            print(f"[reassemble-success-long] preparing {h5_path}")
        summary = run_reassemble_eval_pipeline(
            h5_path=Path(h5_path),
            output_dir=output_dir,
            top_k=args.top_k,
            segment_level="low",
            config_path=args.config_path,
            robot_downsample_hz=args.robot_downsample_hz,
            camera_names=args.camera_names,
            include_event_camera=False,
            extract_frames=True,
            frame_cache_dir=args.frame_cache_dir,
            selectors=args.selectors,
            metric_ranking=args.metric_ranking,
            evaluation_unit="successful_high_level_action",
            max_demos=len(selected),
            min_demo_duration_sec=args.min_duration_sec,
            max_demo_duration_sec=args.max_duration_sec,
            target_demo_bundle_ids=selected_bundle_ids,
            frame_time_windows_by_h5_path={
                str(Path(h5_path).expanduser().resolve()): windows_by_h5[h5_path]
            },
            plot_ecod_aggregates=True,
            resume=args.resume or summary is not None,
            verbose=args.verbose,
        )

    if not args.skip_trimmed_videos:
        export_summary = export_trimmed_videos(output_dir, cameras=args.camera_names)
        (output_dir / "trimmed_video_summary.json").write_text(
            json.dumps(export_summary, indent=2) + "\n",
            encoding="utf-8",
        )

    run_summary_path = output_dir / "long_success_run_summary.json"
    run_summary = {
        "status": "complete",
        "output_dir": str(output_dir),
        "eligible_candidate_count": len(candidates),
        "selected_demo_count": len(selected),
        "balanced_count_per_kind": args.balanced_count_per_kind,
        "task_kinds": args.task_kinds,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "selection": args.selection,
        "include_no_action": args.include_no_action,
        "selected_demos_path": str(output_dir / "selected_long_success_demos.json"),
        "selected_demos_csv_path": str(output_dir / "selected_long_success_demos.csv"),
        "trimmed_videos_dir": str(output_dir / "trimmed_videos"),
        "viewer_summary": summary,
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2))
    print()
    print(
        "Open the viewer with: "
        f"streamlit run pathologist/reassemble/viewer_app.py -- --output-dir {output_dir}"
    )


def scan_success_candidates(
    *,
    data_root: Path,
    min_duration_sec: float,
    max_duration_sec: float,
    include_no_action: bool,
    task_kinds: set[str] | None = None,
) -> list[SuccessCandidate]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Long REASSEMBLE prep requires h5py.") from exc

    candidates: list[SuccessCandidate] = []
    for h5_path in sorted(data_root.glob("*.h5")):
        with h5py.File(h5_path, "r") as handle:
            timestamp_origin = _global_timestamp_origin(handle)
            segments = [segment.to_dict() for segment in _parse_segments(handle, timestamp_origin)]
        low_segments_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for segment in segments:
            if segment.get("level") == "low":
                low_segments_by_parent[str(segment.get("parent_segment_id"))].append(segment)
        for segment in segments:
            if segment.get("level") != "high" or not bool(segment.get("success", True)):
                continue
            low_segments = low_segments_by_parent.get(str(segment.get("segment_id")), [])
            if any(not bool(low_segment.get("success", True)) for low_segment in low_segments):
                continue
            text = str(segment.get("text") or "")
            if not include_no_action and text.strip().lower() == "no action.":
                continue
            task_kind = _task_kind(text)
            if task_kinds is not None and task_kind not in task_kinds:
                continue
            start_sec = float(segment["start_sec"])
            end_sec = float(segment["end_sec"])
            duration_sec = max(0.0, end_sec - start_sec)
            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                continue
            source_file_bundle_id = h5_path.stem
            segment_id = str(segment.get("segment_id", "unknown"))
            candidates.append(
                SuccessCandidate(
                    bundle_id=f"{source_file_bundle_id}_high_{_safe_identifier(segment_id)}",
                    h5_path=str(h5_path.resolve()),
                    source_file_bundle_id=source_file_bundle_id,
                    segment_id=segment_id,
                    segment_index=segment.get("index"),
                    task_text=text,
                    task_kind=task_kind,
                    start_sec_in_file=start_sec,
                    end_sec_in_file=end_sec,
                    duration_sec=duration_sec,
                )
            )
    return candidates


def select_candidates(
    candidates: list[SuccessCandidate],
    *,
    max_demos: int,
    selection: str,
) -> list[SuccessCandidate]:
    if selection == "longest":
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.duration_sec,
                candidate.h5_path,
                int(candidate.segment_index or 0),
            ),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.h5_path,
                candidate.start_sec_in_file,
                int(candidate.segment_index or 0),
            ),
        )
    return ordered[:max_demos]


def select_balanced_candidates(
    candidates: list[SuccessCandidate],
    *,
    task_kinds: list[str],
    count_per_kind: int,
    selection: str,
) -> list[SuccessCandidate]:
    if count_per_kind <= 0:
        raise ValueError("balanced count per kind must be positive.")
    selected: list[SuccessCandidate] = []
    for task_kind in task_kinds:
        kind_candidates = [
            candidate for candidate in candidates if candidate.task_kind == task_kind
        ]
        selected.extend(
            select_candidates(
                kind_candidates,
                max_demos=count_per_kind,
                selection=selection,
            )
        )
    return selected


def export_trimmed_videos(output_dir: Path, *, cameras: list[str]) -> dict[str, Any]:
    manifest_path = output_dir / "reassemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    included = [entry for entry in manifest if entry.get("included") is True]
    output_root = output_dir / "trimmed_videos"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in included:
        bundle_id = str(entry["bundle_id"])
        source_file_bundle_id = str(entry.get("source_file_bundle_id") or Path(entry["h5_path"]).stem)
        start_sec = float(entry["high_level_segment_start_sec_in_file"])
        end_sec = float(entry["high_level_segment_end_sec_in_file"])
        bundle_video_dir = output_root / bundle_id
        bundle_video_dir.mkdir(parents=True, exist_ok=True)
        camera_outputs: dict[str, Any] = {}
        frame_lists: dict[str, list[Path]] = {}
        for camera_name in cameras:
            frame_dir = output_dir / "frame_cache" / source_file_bundle_id / "images" / camera_name
            frames = _frames_in_window(frame_dir, start_sec, end_sec)
            frame_lists[camera_name] = frames
            if not frames:
                camera_outputs[camera_name] = {"frame_count": 0, "video_path": None}
                continue
            video_path = bundle_video_dir / f"{camera_name}.mp4"
            camera_outputs[camera_name] = _write_video_from_frames(
                frames,
                video_path,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        combined_path = bundle_video_dir / "combined_cameras.mp4"
        combined_output = _write_combined_video(
            frame_lists,
            combined_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        rows.append(
            {
                "bundle_id": bundle_id,
                "task_text": (entry.get("high_level_segment") or {}).get("text"),
                "duration_sec": entry.get("duration_sec"),
                "start_sec_in_file": start_sec,
                "end_sec_in_file": end_sec,
                "camera_videos": camera_outputs,
                "combined_video": combined_output,
            }
        )
    return {"trimmed_video_count": len(rows), "videos": rows}


def _write_candidate_outputs(
    output_dir: Path,
    candidates: list[SuccessCandidate],
    selected: list[SuccessCandidate],
) -> None:
    (output_dir / "eligible_long_success_candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2) + "\n",
        encoding="utf-8",
    )
    selected_payload = [asdict(candidate) for candidate in selected]
    (output_dir / "selected_long_success_demos.json").write_text(
        json.dumps(selected_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "selected_long_success_demos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_payload[0]) if selected_payload else [])
        if selected_payload:
            writer.writeheader()
            writer.writerows(selected_payload)


def _frames_in_window(frame_dir: Path, start_sec: float, end_sec: float) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    frames = []
    for image_path in frame_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        timestamp_sec = _timestamp_from_frame_name(image_path)
        if timestamp_sec is None or start_sec <= timestamp_sec <= end_sec:
            frames.append(image_path)
    return sorted(frames, key=_natural_frame_key)


def _write_video_from_frames(
    frames: list[Path],
    video_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    output_fps: float = 30.0,
) -> dict[str, Any]:
    cv2 = _import_cv2()
    first = cv2.imread(str(frames[0]))
    if first is None:
        return {"frame_count": 0, "video_path": None, "error": f"Could not read {frames[0]}"}
    height, width = first.shape[:2]
    fps = float(output_fps)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    written = 0
    timestamped_frames = _timestamped_frames(frames)
    target_times = _render_timestamps(start_sec, end_sec, fps)
    if not timestamped_frames or not target_times:
        writer.release()
        return {"frame_count": 0, "video_path": None, "error": "No timestamped frames in clip window"}
    for target_sec in target_times:
        frame_path = _nearest_timestamped_frame(timestamped_frames, target_sec)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1
    writer.release()
    return {
        "frame_count": written,
        "fps": fps,
        "video_path": str(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": max(0.0, end_sec - start_sec),
        "source_frame_count": len(frames),
    }


def _write_combined_video(
    frame_lists: dict[str, list[Path]],
    video_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    output_fps: float = 30.0,
) -> dict[str, Any]:
    non_empty = [(camera, frames) for camera, frames in frame_lists.items() if frames]
    if not non_empty:
        return {"frame_count": 0, "video_path": None}
    cv2 = _import_cv2()
    fps = float(output_fps)
    target_times = _render_timestamps(start_sec, end_sec, fps)
    if not target_times:
        return {"frame_count": 0, "video_path": None}
    panels = []
    for camera_name, frames in non_empty:
        timestamped_frames = _timestamped_frames(frames)
        if not timestamped_frames:
            continue
        frame = cv2.imread(str(timestamped_frames[0][1]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        target_height = 360
        target_width = max(1, round(width * target_height / max(height, 1)))
        panels.append((camera_name, timestamped_frames, target_width, target_height))
    if not panels:
        return {"frame_count": 0, "video_path": None}
    output_width = sum(panel[2] for panel in panels)
    output_height = max(panel[3] for panel in panels)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    written = 0
    for target_sec in target_times:
        resized_frames = []
        for camera_name, timestamped_frames, width, height in panels:
            frame_path = _nearest_timestamped_frame(timestamped_frames, target_sec)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            frame = cv2.resize(frame, (width, height))
            cv2.putText(
                frame,
                camera_name,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            resized_frames.append(frame)
        if len(resized_frames) != len(panels):
            continue
        writer.write(cv2.hconcat(resized_frames))
        written += 1
    writer.release()
    return {
        "frame_count": written,
        "fps": fps,
        "video_path": str(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": max(0.0, end_sec - start_sec),
        "source_frame_counts": {
            camera_name: len(timestamped_frames)
            for camera_name, timestamped_frames, _, _ in panels
        },
    }


def _timestamped_frames(frames: list[Path]) -> list[tuple[float, Path]]:
    timestamped = [
        (timestamp, frame)
        for frame in frames
        if (timestamp := _timestamp_from_frame_name(frame)) is not None
    ]
    return sorted(timestamped, key=lambda item: item[0])


def _render_timestamps(start_sec: float, end_sec: float, fps: float) -> list[float]:
    duration_sec = max(0.0, float(end_sec) - float(start_sec))
    if duration_sec <= 0 or fps <= 0:
        return []
    frame_count = max(1, int(round(duration_sec * fps)))
    return [float(start_sec) + index / fps for index in range(frame_count)]


def _nearest_timestamped_frame(timestamped_frames: list[tuple[float, Path]], target_sec: float) -> Path:
    import bisect

    timestamps = [timestamp for timestamp, _ in timestamped_frames]
    index = bisect.bisect_left(timestamps, target_sec)
    if index <= 0:
        return timestamped_frames[0][1]
    if index >= len(timestamped_frames):
        return timestamped_frames[-1][1]
    before = timestamped_frames[index - 1]
    after = timestamped_frames[index]
    return before[1] if abs(before[0] - target_sec) <= abs(after[0] - target_sec) else after[1]


def _fps_from_frame_names(frames: list[Path]) -> float:
    timestamps = [
        timestamp
        for frame in frames
        if (timestamp := _timestamp_from_frame_name(frame)) is not None
    ]
    if len(timestamps) >= 2:
        diffs = [
            later - earlier
            for earlier, later in zip(timestamps, timestamps[1:])
            if later > earlier
        ]
        if diffs:
            return max(1.0, min(60.0, 1.0 / median(diffs)))
    return 30.0


def _timestamp_from_frame_name(path: Path) -> float | None:
    match = re.search(r"_t(\d+(?:\.\d+)?)", path.stem)
    if match is None:
        return None
    return float(match.group(1))


def _natural_frame_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    key.append(path.suffix.lower())
    return key


def _safe_identifier(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "unknown"


def _task_kind(text: str) -> str:
    normalized = text.strip().lower()
    if not normalized:
        return "unknown"
    return normalized.split()[0].strip(".")


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Trimmed REASSEMBLE video export requires opencv-python/cv2.") from exc
    return cv2


if __name__ == "__main__":
    main()
