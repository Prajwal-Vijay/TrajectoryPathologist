"""Synchronized three-view ABC mosaic export with clock sidecars."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

from VL_implemenation.abc_data import (
    Episode,
    camera_cache_directory,
    camera_timeline,
)


CAMERA_NAMES = ("top", "left_wrist", "right_wrist")


def mosaic_paths(mosaic_root: Path, episode: Episode) -> dict[str, Path]:
    directory = mosaic_root / episode.task_name / episode.bundle_id
    return {
        "directory": directory,
        "video": directory / "mosaic_3view.mp4",
        "timeline": directory / "mosaic_timeline.jsonl",
        "metadata": directory / "mosaic_export.json",
    }


def export_episode_mosaic(
    episode: Episode,
    *,
    frame_cache_root: Path,
    mosaic_root: Path,
    camera_sample_hz: float,
    panel_height: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = mosaic_paths(mosaic_root, episode)
    if not overwrite and _complete_export(paths):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    timelines = {
        camera_name: camera_timeline(
            camera_cache_directory(
                frame_cache_root,
                episode,
                camera_name,
                camera_sample_hz,
            )
        )
        for camera_name in CAMERA_NAMES
    }
    records = build_mosaic_timeline(timelines, output_fps=camera_sample_hz)
    if not records:
        raise ValueError(f"No mosaic records for {episode.bundle_id}")

    paths["directory"].mkdir(parents=True, exist_ok=True)
    partial_video = paths["directory"] / "mosaic_3view.partial.mp4"
    if partial_video.exists():
        partial_video.unlink()
    video_details = _write_video(
        records,
        partial_video,
        fps=camera_sample_hz,
        panel_height=panel_height,
    )
    partial_video.replace(paths["video"])
    _write_jsonl_atomic(paths["timeline"], records)

    max_sync_delta = {
        camera_name: max(
            float(record[f"{camera_name}_delta_sec"]) for record in records
        )
        for camera_name in ("left_wrist", "right_wrist")
    }
    metadata = {
        "status": "complete",
        "task_name": episode.task_name,
        "bundle_id": episode.bundle_id,
        "video_path": str(paths["video"]),
        "timeline_path": str(paths["timeline"]),
        "frame_count": len(records),
        "output_fps": camera_sample_hz,
        "video_duration_sec": len(records) / camera_sample_hz,
        "first_trajectory_sec": records[0]["trajectory_sec"],
        "last_trajectory_sec": records[-1]["trajectory_sec"],
        "trajectory_duration_sec": episode.duration_sec,
        "sync_policy": "top_master_nearest_wrist",
        "camera_order": list(CAMERA_NAMES),
        "source_frame_counts": {
            name: len(timeline) for name, timeline in timelines.items()
        },
        "max_sync_delta_sec": max_sync_delta,
        "panel_height": panel_height,
        **video_details,
    }
    _write_json_atomic(paths["metadata"], metadata)
    return metadata


def build_mosaic_timeline(
    timelines: dict[str, list[tuple[float, Path]]],
    *,
    output_fps: float,
) -> list[dict[str, Any]]:
    top = timelines["top"]
    wrists = {
        camera_name: timelines[camera_name]
        for camera_name in ("left_wrist", "right_wrist")
    }
    output: list[dict[str, Any]] = []
    for frame_index, (trajectory_sec, top_path) in enumerate(top):
        record: dict[str, Any] = {
            "frame_index": frame_index,
            "video_sec": frame_index / output_fps,
            "trajectory_sec": trajectory_sec,
            "top_path": str(top_path),
            "top_src_sec": trajectory_sec,
            "top_delta_sec": 0.0,
        }
        for camera_name, timeline in wrists.items():
            source_sec, source_path = nearest_timestamped_frame(
                timeline, trajectory_sec
            )
            record[f"{camera_name}_path"] = str(source_path)
            record[f"{camera_name}_src_sec"] = source_sec
            record[f"{camera_name}_delta_sec"] = abs(source_sec - trajectory_sec)
        output.append(record)
    return output


def nearest_timestamped_frame(
    timeline: list[tuple[float, Path]], target_sec: float
) -> tuple[float, Path]:
    timestamps = [item[0] for item in timeline]
    index = bisect.bisect_left(timestamps, target_sec)
    if index <= 0:
        return timeline[0]
    if index >= len(timeline):
        return timeline[-1]
    before = timeline[index - 1]
    after = timeline[index]
    return before if abs(before[0] - target_sec) <= abs(after[0] - target_sec) else after


def load_mosaic_timeline(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Empty mosaic timeline: {path}")
    return records


def video_sec_to_trajectory_sec(
    records: list[dict[str, Any]], video_sec: float
) -> float:
    video_times = [float(record["video_sec"]) for record in records]
    index = bisect.bisect_left(video_times, video_sec)
    if index <= 0:
        return float(records[0]["trajectory_sec"])
    if index >= len(records):
        return float(records[-1]["trajectory_sec"])
    left = records[index - 1]
    right = records[index]
    left_video = float(left["video_sec"])
    right_video = float(right["video_sec"])
    if right_video <= left_video:
        return float(left["trajectory_sec"])
    fraction = (video_sec - left_video) / (right_video - left_video)
    return float(left["trajectory_sec"]) + fraction * (
        float(right["trajectory_sec"]) - float(left["trajectory_sec"])
    )


def _write_video(
    records: list[dict[str, Any]],
    path: Path,
    *,
    fps: float,
    panel_height: int,
) -> dict[str, Any]:
    import cv2

    first_frames = [
        cv2.imread(records[0][f"{camera_name}_path"])
        for camera_name in CAMERA_NAMES
    ]
    if any(frame is None for frame in first_frames):
        raise RuntimeError("Could not read one or more first mosaic frames")
    panel_widths = []
    for frame in first_frames:
        height, width = frame.shape[:2]
        target_width = max(2, round(width * panel_height / max(height, 1)))
        if target_width % 2:
            target_width += 1
        panel_widths.append(target_width)
    output_width = sum(panel_widths)
    if output_width % 2:
        output_width += 1

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (output_width, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    written = 0
    try:
        for record in records:
            panels = []
            for camera_name, panel_width in zip(CAMERA_NAMES, panel_widths):
                frame = cv2.imread(record[f"{camera_name}_path"])
                if frame is None:
                    raise RuntimeError(
                        f"Could not read {record[f'{camera_name}_path']}"
                    )
                panel = cv2.resize(
                    frame,
                    (panel_width, panel_height),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.putText(
                    panel,
                    camera_name,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            mosaic = cv2.hconcat(panels)
            if mosaic.shape[1] != output_width:
                mosaic = cv2.copyMakeBorder(
                    mosaic,
                    0,
                    0,
                    0,
                    output_width - mosaic.shape[1],
                    cv2.BORDER_CONSTANT,
                )
            writer.write(mosaic)
            written += 1
    finally:
        writer.release()
    if written != len(records):
        raise RuntimeError(f"Wrote {written}/{len(records)} frames to {path}")
    return {
        "codec": "mp4v",
        "output_width": output_width,
        "output_height": panel_height,
    }


def _complete_export(paths: dict[str, Path]) -> bool:
    if not all(paths[name].is_file() for name in ("video", "timeline", "metadata")):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        metadata.get("status") == "complete"
        and paths["video"].stat().st_size > 0
        and int(metadata.get("frame_count", 0)) > 0
    )


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
