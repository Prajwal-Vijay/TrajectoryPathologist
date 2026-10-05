"""ABC-130K discovery and insertion/removal ground-truth helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FRAME_TIMESTAMP_RE = re.compile(r"_t(\d+(?:\.\d+)?)")


@dataclass(frozen=True, slots=True)
class GroundInterval:
    event_type: str
    start_sec: float
    end_sec: float
    annotation_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Episode:
    task_name: str
    bundle_id: str
    episode_path: Path
    annotation_path: Path
    duration_sec: float
    ground_intervals: tuple[GroundInterval, ...]

    @property
    def instruction(self) -> str:
        return self.task_name.replace("_", " ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "bundle_id": self.bundle_id,
            "episode_path": str(self.episode_path),
            "annotation_path": str(self.annotation_path),
            "duration_sec": self.duration_sec,
            "instruction": self.instruction,
            "ground_intervals": [
                interval.to_dict() for interval in self.ground_intervals
            ],
        }


def semantic_event_type(annotation_text: str) -> str | None:
    """Map the predeclared ABC semantic aliases to an event type."""

    normalized = " ".join(annotation_text.strip().casefold().split())
    if normalized.startswith("insert ") or normalized.startswith(
        "press the plug into "
    ):
        return "insertion"
    if normalized.startswith("remove ") or normalized.startswith("take out "):
        return "removal"
    return None


def instruction_event_type(instruction: str) -> str | None:
    """Infer an explicit event type from the task instruction only."""

    normalized = " ".join(instruction.strip().casefold().split())
    if normalized.startswith("insert ") or normalized.startswith("plug in "):
        return "insertion"
    if normalized.startswith("remove ") or normalized.startswith("take out "):
        return "removal"
    return None


def discover_episodes(data_root: str | Path) -> list[Episode]:
    output: list[Episode] = []
    root = Path(data_root).expanduser().resolve()
    from mcap.reader import make_reader

    for episode_path in sorted(root.glob("*/*/episode.mcap")):
        annotation_path = episode_path.with_name("annotation.mcap")
        if not annotation_path.is_file():
            continue
        with episode_path.open("rb") as handle:
            summary = make_reader(handle).get_summary()
        if summary is None or summary.statistics is None:
            continue
        origin_ns = int(summary.statistics.message_start_time)
        end_ns = int(summary.statistics.message_end_time)
        annotated_segments = _load_annotation_segments(
            annotation_path,
            origin_ns=origin_ns,
            end_ns=end_ns,
        )
        intervals = tuple(
            GroundInterval(
                event_type=event_type,
                start_sec=start_sec,
                end_sec=end_sec,
                annotation_text=text,
            )
            for start_sec, end_sec, text in annotated_segments
            if (event_type := semantic_event_type(text)) is not None
        )
        output.append(
            Episode(
                task_name=episode_path.parent.parent.name,
                bundle_id=episode_path.parent.name,
                episode_path=episode_path,
                annotation_path=annotation_path,
                duration_sec=max(0.0, (end_ns - origin_ns) / 1e9),
                ground_intervals=intervals,
            )
        )
    return output


def _load_annotation_segments(
    annotation_path: Path,
    *,
    origin_ns: int,
    end_ns: int,
) -> list[tuple[float, float, str]]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    annotations: list[tuple[int, str]] = []
    with annotation_path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded in reader.iter_decoded_messages(
            topics=["/subtask-annotation"]
        ):
            timestamp_ns = _message_timestamp_ns(decoded, int(message.log_time))
            text = str(getattr(decoded, "data", "")).strip()
            if text:
                annotations.append((timestamp_ns, text))
    annotations.sort(key=lambda item: item[0])
    output: list[tuple[float, float, str]] = []
    for index, (start_ns, text) in enumerate(annotations):
        segment_end_ns = (
            annotations[index + 1][0] if index + 1 < len(annotations) else end_ns
        )
        clipped_start = min(max(start_ns, origin_ns), end_ns)
        clipped_end = min(max(segment_end_ns, clipped_start), end_ns)
        output.append(
            (
                (clipped_start - origin_ns) / 1e9,
                (clipped_end - origin_ns) / 1e9,
                text,
            )
        )
    return output


def _message_timestamp_ns(decoded: Any, fallback_ns: int) -> int:
    timestamp = getattr(decoded, "timestamp", None)
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if seconds is None or nanos is None:
        return fallback_ns
    return int(seconds) * 1_000_000_000 + int(nanos)


def timestamp_from_frame_path(path: Path) -> float:
    match = FRAME_TIMESTAMP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Frame filename has no _t timestamp: {path}")
    return float(match.group(1))


def camera_timeline(directory: Path) -> list[tuple[float, Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing camera cache directory: {directory}")
    timeline = [
        (timestamp_from_frame_path(path), path.resolve())
        for path in directory.glob("*.jpg")
    ]
    timeline.sort(key=lambda item: (item[0], item[1].name))
    if not timeline:
        raise ValueError(f"No JPEG frames found in {directory}")
    return timeline


def camera_cache_directory(
    frame_cache_root: Path,
    episode: Episode,
    camera_name: str,
    sample_hz: float,
) -> Path:
    return (
        frame_cache_root
        / episode.task_name
        / episode.bundle_id
        / f"{camera_name}_{sample_hz:g}hz"
    )
