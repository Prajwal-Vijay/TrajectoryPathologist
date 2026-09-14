"""Classify REASSEMBLE selected peaks into event types and score against hand annotations.

Each selector already chose its top-K peaks per demo, so this runner only asks the VLM
what event is happening at each selected frame, then matches the prediction against the
nearest hand-annotated event. The comparison is only fair if every selector gets the
identical prompt, frame count, and matching rule, so all three share one code path.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.qa.reassemble_events import (
    DEFAULT_CAMERAS,
    DEFAULT_TOLERANCES_SEC,
    EVENT_LABELS,
    SELECTOR_ALIASES,
    SELECTOR_NAMES,
    aggregate_event_stats,
    build_event_classification_prompt,
    parse_event_annotations,
    parse_event_prediction,
    resolve_selector_names,
)
from pathologist.qa.robofac_client import (
    OpenAIRoboFACQAClient,
    RoboFACQAClient,
    StaticRoboFACQAClient,
)
from pathologist.schemas.trajectory_bundle import CameraFrameRef


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PEAKS_DIR = REPO_ROOT / "REASSEMBLE_Successful_Balanced"
FRAME_TIMESTAMP_PATTERN = re.compile(r"_t(\d+(?:\.\d+)?)\.")


def run_reassemble_event_qa_pipeline(
    peaks_dir: str | Path = DEFAULT_PEAKS_DIR,
    annotations_path: str | Path | None = None,
    output_dir: str | Path = "reassemble_event_qa_outputs",
    selectors: list[str] | None = None,
    cameras: list[str] | None = None,
    context_frames: int = 1,
    context_offset_sec: float = 0.4,
    tolerance_sec: float = 2.0,
    tolerance_sweep_sec: list[float] | None = None,
    bootstrap_samples: int = 10000,
    score_every: int = 25,
    score_only: bool = False,
    max_demos: int | None = None,
    qa_client: RoboFACQAClient | None = None,
    openai_model: str | None = None,
    openai_temperature: float | None = None,
    openai_timeout_sec: float | None = None,
    openai_max_retries: int | None = None,
    openai_retry_backoff_sec: float | None = None,
    fail_on_vlm_error: bool = False,
    resume: bool = False,
    skip_vlm: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Classify each selector's peaks and score them against the annotated events."""

    if context_frames < 0:
        raise ValueError("context_frames must be non-negative.")
    if context_offset_sec <= 0 and context_frames:
        raise ValueError("context_offset_sec must be positive when context_frames > 0.")
    peaks_path = Path(peaks_dir).expanduser().resolve()
    annotations_file = (
        Path(annotations_path).expanduser().resolve()
        if annotations_path is not None
        else peaks_path / "annotations.txt"
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_selectors = resolve_selector_names(selectors)
    resolved_cameras = [str(name) for name in (cameras or DEFAULT_CAMERAS)]
    if not resolved_cameras:
        raise ValueError("At least one camera is required.")
    resolved_sweep = sorted({float(value) for value in (tolerance_sweep_sec or DEFAULT_TOLERANCES_SEC)})

    predictions_path = output_path / "event_predictions.jsonl"
    stats_path = output_path / "event_stats.json"
    annotations_out_path = output_path / "parsed_annotations.json"
    summary_path = output_path / "run_summary.json"

    events_by_demo, annotation_notes = parse_event_annotations(annotations_file)
    scoreable_demos = {demo_id for demo_id, events in events_by_demo.items() if events}
    demo_metadata = _load_demo_metadata(peaks_path / "reassemble_manifest.json")
    _write_json_atomic(
        annotations_out_path,
        {
            "annotations_path": str(annotations_file),
            "labels": list(EVENT_LABELS),
            "demo_count": len(events_by_demo),
            "scoreable_demo_count": len(scoreable_demos),
            "event_count": sum(len(events) for events in events_by_demo.values()),
            "parse_notes": annotation_notes,
            "events_by_demo": {
                demo_id: [event.to_dict() for event in events]
                for demo_id, events in sorted(events_by_demo.items())
            },
        },
    )

    demo_ids = sorted(scoreable_demos & set(demo_metadata))
    missing_metadata = sorted(scoreable_demos - set(demo_metadata))
    if max_demos is not None:
        demo_ids = demo_ids[: max(0, max_demos)]
    _log(
        verbose,
        f"Annotated {len(scoreable_demos)} demo(s); evaluating {len(demo_ids)} with "
        f"selectors {resolved_selectors}.",
    )
    if missing_metadata:
        _warn(f"{len(missing_metadata)} annotated demo(s) missing from the manifest: {missing_metadata}")

    existing_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, int]] = set()
    if (resume or score_only) and predictions_path.exists():
        existing_rows = _read_jsonl(predictions_path)
        completed_keys = {
            (str(row["demo_id"]), str(row["selector"]), int(row["peak_rank"]))
            for row in existing_rows
            if row.get("raw_response") or row.get("predicted_label")
        }
        _log(verbose, f"Resume: {len(completed_keys)} prediction(s) already on disk.")

    client = qa_client
    if client is None and not score_only:
        client = (
            StaticRoboFACQAClient("")
            if skip_vlm
            else OpenAIRoboFACQAClient(
                model=openai_model,
                temperature=openai_temperature,
                timeout_sec=openai_timeout_sec,
                max_retries=openai_max_retries,
                retry_backoff_sec=openai_retry_backoff_sec,
                max_output_tokens=256,
            )
        )

    def score(rows: list[dict[str, Any]], *, final: bool) -> dict[str, Any]:
        """Aggregate and persist stats. Interim passes skip the bootstrap to stay cheap."""

        stats = aggregate_event_stats(
            rows,
            events_by_demo,
            tolerances_sec=resolved_sweep,
            primary_tolerance_sec=tolerance_sec,
            bootstrap_samples=bootstrap_samples if final else 0,
        )
        stats["is_final"] = final
        _write_json_atomic(stats_path, stats)
        return stats

    frame_index = _FrameIndex(peaks_path)
    rows: list[dict[str, Any]] = list(existing_rows)
    vlm_error_count = 0
    skipped_peaks: list[dict[str, Any]] = []
    for demo_id in [] if score_only else demo_ids:
        metadata = demo_metadata[demo_id]
        selected = _load_selected_peaks(peaks_path / "selected_peaks" / demo_id / "selected_peaks.json")
        for selector in resolved_selectors:
            peaks = selected.get(selector, [])
            if not peaks:
                _warn(f"[{demo_id}] selector={selector} has no selected peaks; skipping.")
                continue
            for peak in peaks:
                peak_rank = int(peak.get("timeline_rank") or 0)
                key = (demo_id, selector, peak_rank)
                if key in completed_keys:
                    continue
                try:
                    frames, frame_groups = _build_query_frames(
                        peak=peak,
                        metadata=metadata,
                        frame_index=frame_index,
                        cameras=resolved_cameras,
                        context_frames=context_frames,
                        context_offset_sec=context_offset_sec,
                    )
                except Exception as exc:
                    skipped_peaks.append(
                        {
                            "demo_id": demo_id,
                            "selector": selector,
                            "peak_rank": peak_rank,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _warn(f"[{demo_id}/{selector}/{peak_rank}] frame lookup failed: {exc}")
                    continue
                prompt = build_event_classification_prompt(
                    task_text=metadata["task_text"],
                    frame_groups=frame_groups,
                )
                raw_response = ""
                vlm_error = None
                try:
                    raw_response = client.answer(prompt, frames)
                except Exception as exc:
                    vlm_error = f"{type(exc).__name__}: {exc}"
                    vlm_error_count += 1
                    _warn(f"[{demo_id}/{selector}/{peak_rank}] VLM error: {vlm_error}")
                    if fail_on_vlm_error:
                        _write_jsonl_atomic(predictions_path, rows)
                        raise
                row = {
                    "demo_id": demo_id,
                    "selector": selector,
                    "peak_rank": peak_rank,
                    "event_id": peak.get("event_id"),
                    "peak_sec": float(peak["peak_sec"]),
                    "task_text": metadata["task_text"],
                    "task_kind": metadata["task_kind"],
                    "cameras": resolved_cameras,
                    "query_frame_paths": [str(frame.image_path) for frame in frames],
                    "query_frame_offsets_sec": {
                        camera_name: offsets for camera_name, offsets in frame_groups
                    },
                    "predicted_label": parse_event_prediction(raw_response),
                    "raw_response": raw_response,
                    "vlm_error": vlm_error,
                }
                rows.append(row)
                completed_keys.add(key)
                _write_jsonl_atomic(predictions_path, rows)
                _log(
                    verbose,
                    f"[{demo_id}/{selector}/{peak_rank}] t={row['peak_sec']:.2f}s "
                    f"-> {row['predicted_label']}",
                )
                if score_every and len(rows) % score_every == 0:
                    interim = score(rows, final=False)
                    _log(verbose, f"Running scores after {len(rows)}: {_format_scores(interim)}")

    stats = score(rows, final=True)
    _write_jsonl_atomic(predictions_path, rows)
    _log(verbose, f"Final scores: {_format_scores(stats)}")

    summary = {
        "status": "complete",
        "peaks_dir": str(peaks_path),
        "annotations_path": str(annotations_file),
        "output_dir": str(output_path),
        "selectors": resolved_selectors,
        "cameras": resolved_cameras,
        "labels": list(EVENT_LABELS),
        "context_frames": context_frames,
        "context_offset_sec": context_offset_sec,
        "frames_per_query": len(resolved_cameras) * (1 + 2 * context_frames),
        "primary_tolerance_sec": tolerance_sec,
        "tolerance_sweep_sec": resolved_sweep,
        "bootstrap_samples": bootstrap_samples,
        "score_every": score_every,
        "score_only": score_only,
        "skip_vlm": skip_vlm,
        "resume": resume,
        "openai_model": None if (skip_vlm or score_only) else getattr(client, "model", openai_model),
        "annotated_demo_count": len(scoreable_demos),
        "evaluated_demo_count": len(demo_ids),
        "demos_missing_from_manifest": missing_metadata,
        "prediction_count": len(rows),
        "vlm_error_count": vlm_error_count,
        "skipped_peak_count": len(skipped_peaks),
        "skipped_peaks": skipped_peaks,
        "annotation_parse_note_count": len(annotation_notes),
        "stats": stats,
    }
    _write_json_atomic(summary_path, summary)
    _log(verbose, f"Wrote {len(rows)} prediction(s) to {predictions_path}.")
    return summary


class _FrameIndex:
    """Nearest-frame lookup over the dense per-session frame cache."""

    def __init__(self, peaks_path: Path) -> None:
        self._peaks_path = peaks_path
        self._cache: dict[tuple[str, str], tuple[list[float], list[Path]]] = {}

    def nearest(self, session_id: str, camera_name: str, target_sec: float) -> tuple[Path, float]:
        timestamps, paths = self._load(session_id, camera_name)
        if not timestamps:
            raise FileNotFoundError(f"No cached frames for {session_id}/{camera_name}.")
        position = bisect.bisect_left(timestamps, target_sec)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
        best = min(candidates, key=lambda index: abs(timestamps[index] - target_sec))
        return paths[best], timestamps[best]

    def _load(self, session_id: str, camera_name: str) -> tuple[list[float], list[Path]]:
        key = (session_id, camera_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        camera_dir = self._peaks_path / "frame_cache" / session_id / "images" / camera_name
        entries: list[tuple[float, Path]] = []
        if camera_dir.is_dir():
            for path in camera_dir.iterdir():
                match = FRAME_TIMESTAMP_PATTERN.search(path.name)
                if match is not None:
                    entries.append((float(match.group(1)), path))
        entries.sort()
        resolved = ([timestamp for timestamp, _ in entries], [path for _, path in entries])
        self._cache[key] = resolved
        return resolved


def _build_query_frames(
    *,
    peak: dict[str, Any],
    metadata: dict[str, Any],
    frame_index: _FrameIndex,
    cameras: list[str],
    context_frames: int,
    context_offset_sec: float,
) -> tuple[list[CameraFrameRef], list[tuple[str, list[float]]]]:
    """Resolve the peak plus symmetric neighbours on each fixed camera.

    Frames come from ``peak_sec`` rather than from any camera recorded alongside the peak,
    because the metric selector stored timestamps only, and resolving all three selectors
    the same way is what keeps the comparison about peak choice. Offsets are recomputed
    from the frames actually found so the prompt never claims a spacing the images lack.
    """

    peak_sec = float(peak["peak_sec"])
    session_id = str(metadata["session_id"])
    demo_start_in_file = float(metadata["start_sec_in_file"])
    demo_end_in_file = float(metadata["end_sec_in_file"])
    peak_in_file = demo_start_in_file + peak_sec

    frames: list[CameraFrameRef] = []
    groups: list[tuple[str, list[float]]] = []
    for camera_name in cameras:
        peak_path, peak_actual_sec = frame_index.nearest(session_id, camera_name, peak_in_file)
        collected: dict[Path, float] = {peak_path: peak_actual_sec}
        for step in range(1, context_frames + 1):
            for sign in (-1, 1):
                target = peak_in_file + sign * step * context_offset_sec
                clamped = min(max(target, demo_start_in_file), demo_end_in_file)
                path, actual_sec = frame_index.nearest(session_id, camera_name, clamped)
                collected.setdefault(path, actual_sec)
        offsets: list[float] = []
        for path, actual_sec in sorted(collected.items(), key=lambda item: item[1]):
            frames.append(
                CameraFrameRef(
                    camera_name=camera_name,
                    timestamp_sec=max(0.0, actual_sec - demo_start_in_file),
                    image_path=path,
                    frame_index=None,
                )
            )
            # The frame nearest the peak is the query frame by construction; its sub-frame
            # offset is reported as 0 so the prompt can point at it unambiguously.
            offsets.append(0.0 if path == peak_path else round(actual_sec - peak_in_file, 3))
        groups.append((camera_name, offsets))
    return frames, groups


def _load_demo_metadata(manifest_path: Path) -> dict[str, dict[str, Any]]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not entry.get("included", True):
            continue
        segment = entry.get("high_level_segment") or {}
        metadata[str(entry["bundle_id"])] = {
            "session_id": str(entry.get("source_file_bundle_id") or ""),
            "task_text": str(segment.get("text") or entry.get("high_level_task_kind") or "unknown"),
            "task_kind": entry.get("high_level_task_kind"),
            "duration_sec": entry.get("duration_sec"),
            "start_sec_in_file": entry.get("high_level_segment_start_sec_in_file"),
            "end_sec_in_file": entry.get("high_level_segment_end_sec_in_file"),
            "peaks_dir": str(manifest_path.parent),
        }
    return metadata


def _load_selected_peaks(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_selector = payload.get("selected_peaks_by_selector") or {}
    return {str(name): list(peaks) for name, peaks in by_selector.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _format_scores(stats: dict[str, Any]) -> str:
    """One-line per-selector summary at the primary tolerance, for progress logging."""

    parts = []
    for selector, selector_stats in stats.get("by_selector", {}).items():
        primary = selector_stats.get("primary", {})
        matched = primary.get("matched_frame_count", 0)
        conditional = primary.get("conditional_accuracy")
        end_to_end = primary.get("end_to_end_accuracy")
        conditional_text = "n/a" if conditional is None else f"{conditional * 100:.1f}%"
        end_to_end_text = "n/a" if end_to_end is None else f"{end_to_end * 100:.1f}%"
        parts.append(
            f"{selector} cond={conditional_text} e2e={end_to_end_text} "
            f"(n={primary.get('frame_count', 0)}, matched={matched})"
        )
    return " | ".join(parts) if parts else "no predictions yet"


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[reassemble-events] {message}", flush=True)


def _warn(message: str) -> None:
    print(f"[reassemble-events][warn] {message}", file=sys.stderr, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify REASSEMBLE selected peaks into event types and score them."
    )
    parser.add_argument(
        "--peaks-dir",
        default=str(DEFAULT_PEAKS_DIR),
        help="Directory holding selected_peaks/, frame_cache/, and reassemble_manifest.json.",
    )
    parser.add_argument("--annotations", help="Annotation file. Defaults to <peaks-dir>/annotations.txt.")
    parser.add_argument("--output-dir", default="reassemble_event_qa_outputs", help="Output directory.")
    parser.add_argument(
        "--selectors",
        nargs="+",
        metavar="NAME",
        default=list(SELECTOR_NAMES),
        help=(
            "Selectors to compare, space- or comma-separated. "
            f"Choose from all, {', '.join(SELECTOR_NAMES)} "
            f"(aliases: {', '.join(sorted(SELECTOR_ALIASES))}). Defaults to all."
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        metavar="NAME",
        default=list(DEFAULT_CAMERAS),
        help=(
            "Cameras used for every peak of every selector, held fixed so camera choice "
            f"cannot masquerade as a selector difference. Defaults to {' '.join(DEFAULT_CAMERAS)}."
        ),
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=1,
        help="Neighbour frames per side around each peak. 0 sends the peak frame alone.",
    )
    parser.add_argument(
        "--context-offset-sec",
        type=float,
        default=0.4,
        help="Spacing between the peak frame and each neighbour.",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=2.0,
        help="Headline tolerance for matching a selected frame to the nearest annotated event.",
    )
    parser.add_argument(
        "--tolerance-sweep-sec",
        nargs="+",
        type=float,
        default=list(DEFAULT_TOLERANCES_SEC),
        help="Additional tolerances to report, so the headline number is not a lone choice.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Clustered bootstrap resamples (clustered by demo) for 95%% CIs. 0 disables.",
    )
    parser.add_argument(
        "--score-every",
        type=int,
        default=25,
        help=(
            "Rewrite event_stats.json every N predictions so progress is visible mid-run. "
            "Interim passes skip the bootstrap. 0 scores only at the end."
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score the predictions already in --output-dir without making any VLM calls.",
    )
    parser.add_argument("--max-demos", type=int, help="Optional cap on annotated demos.")
    parser.add_argument("--openai-model", help="OpenAI model for classification calls.")
    parser.add_argument("--openai-temperature", type=float, help="OpenAI sampling temperature.")
    parser.add_argument("--openai-timeout-sec", type=float, help="Per-request OpenAI timeout seconds.")
    parser.add_argument("--openai-max-retries", type=int, help="Retries after the initial request.")
    parser.add_argument("--openai-retry-backoff-sec", type=float, help="Initial retry backoff seconds.")
    parser.add_argument(
        "--fail-on-vlm-error",
        action="store_true",
        help="Stop on the first VLM failure instead of recording it and continuing.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse predictions already in --output-dir.")
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Build frames and prompts without calling the VLM.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        selectors = resolve_selector_names(args.selectors)
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_reassemble_event_qa_pipeline(
        peaks_dir=args.peaks_dir,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        selectors=selectors,
        cameras=args.cameras,
        context_frames=args.context_frames,
        context_offset_sec=args.context_offset_sec,
        tolerance_sec=args.tolerance_sec,
        tolerance_sweep_sec=args.tolerance_sweep_sec,
        bootstrap_samples=args.bootstrap_samples,
        score_every=args.score_every,
        score_only=args.score_only,
        max_demos=args.max_demos,
        openai_model=args.openai_model,
        openai_temperature=args.openai_temperature,
        openai_timeout_sec=args.openai_timeout_sec,
        openai_max_retries=args.openai_max_retries,
        openai_retry_backoff_sec=args.openai_retry_backoff_sec,
        fail_on_vlm_error=args.fail_on_vlm_error,
        resume=args.resume,
        skip_vlm=args.skip_vlm,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
