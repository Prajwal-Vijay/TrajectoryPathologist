"""Frame selectors for RoboFAC QA comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


def select_keyframes(
    selector_name: str,
    bundle: TrajectoryBundle,
    frame_budget: int,
    metric_profile: Any | None = None,
    config: dict[str, Any] | None = None,
    ecod_aggregate_path: str | Path | None = None,
    selection_stats: dict[str, Any] | None = None,
) -> list[CameraFrameRef]:
    """Select a fixed-budget frame set using a named strategy."""

    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive.")
    selector_name = selector_name.lower()
    if selector_name == "uniform":
        return select_uniform_keyframes(bundle.camera_frames, frame_budget)
    if selector_name in {"flow", "kite_flow"}:
        return select_kite_flow_keyframes(
            bundle.camera_frames,
            frame_budget,
            config=config or {},
            pad_to_budget=True,
        )
    if selector_name == "metric":
        resolved_profile = metric_profile or MetricEngine(config=config or {}).compute(bundle)
        return select_metric_keyframes(
            bundle,
            resolved_profile,
            frame_budget,
            config=config or {},
            ecod_aggregate_path=ecod_aggregate_path,
            selection_stats=selection_stats,
        )
    raise ValueError(f"Unsupported RoboFAC selector: {selector_name}")


def select_uniform_keyframes(frames: list[CameraFrameRef], frame_budget: int) -> list[CameraFrameRef]:
    """Select uniformly spaced frames."""

    frames = sorted(frames, key=_frame_sort_key)
    if len(frames) <= frame_budget:
        return frames
    if frame_budget == 1:
        return [frames[len(frames) // 2]]
    indices = [
        round(index * (len(frames) - 1) / (frame_budget - 1))
        for index in range(frame_budget)
    ]
    return [frames[index] for index in sorted(set(indices))][:frame_budget]


def select_metric_keyframes(
    bundle: TrajectoryBundle,
    metric_profile: Any,
    frame_budget: int,
    config: dict[str, Any],
    ecod_aggregate_path: str | Path | None = None,
    selection_stats: dict[str, Any] | None = None,
) -> list[CameraFrameRef]:
    """Select frames from metric-overlap proposals without exposing metric values.

    When the surprise landscape yields fewer than ``frame_budget`` distinct peaks the
    remainder is padded with uniform frames. Pass ``selection_stats`` to observe how
    many frames came from each source instead of letting the fallback pass silently.
    """

    selector_config = config.get("selector") if isinstance(config.get("selector"), dict) else {}
    surprise_flood_config = selector_config.get("surprise_flood", {})
    if not isinstance(surprise_flood_config, dict):
        surprise_flood_config = {}
    selector = MetricEventSelector(
        frame_stride_within_window=int(selector_config.get("frame_stride_within_window", 1)),
        visual_context_sec=float(selector_config.get("visual_context_sec", 0.5)),
        max_frames_per_camera=max(frame_budget, int(selector_config.get("max_frames_per_camera", frame_budget))),
        metric_packet_context_steps=0,
        surprise_flood_config=surprise_flood_config,
    )
    proposals = selector.select(
        metric_profile,
        trajectory_bundle=bundle,
        ecod_aggregate_path=ecod_aggregate_path,
    )
    chosen: list[CameraFrameRef] = []
    seen: set[tuple[str, int | None, float]] = set()
    for proposal in proposals:
        frame = _representative_frame_for_proposal(
            proposal.candidate_frames,
            proposal.time_window,
            proposal=proposal,
            bundle_frames=bundle.camera_frames,
        )
        if frame is None:
            continue
        key = (frame.camera_name, frame.frame_index, frame.timestamp_sec)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(frame)
        if len(chosen) >= frame_budget:
            _record_selection_stats(
                selection_stats,
                proposal_count=len(proposals),
                metric_frame_count=len(chosen),
                frame_budget=frame_budget,
                selected_frame_count=len(chosen),
            )
            return sorted(chosen, key=_frame_sort_key)
    metric_frame_count = len(chosen)
    if len(chosen) < frame_budget:
        chosen = _uniformly_pad_keyframes(
            chosen,
            bundle.camera_frames,
            frame_budget,
        )
    _record_selection_stats(
        selection_stats,
        proposal_count=len(proposals),
        metric_frame_count=metric_frame_count,
        frame_budget=frame_budget,
        selected_frame_count=len(chosen),
    )
    return sorted(chosen[:frame_budget], key=_frame_sort_key)


def _record_selection_stats(
    selection_stats: dict[str, Any] | None,
    *,
    proposal_count: int,
    metric_frame_count: int,
    frame_budget: int,
    selected_frame_count: int,
) -> None:
    if selection_stats is None:
        return
    padded = max(0, selected_frame_count - metric_frame_count)
    selection_stats.update(
        {
            "proposal_count": proposal_count,
            "metric_frame_count": metric_frame_count,
            "uniform_padded_frame_count": padded,
            "frame_budget": frame_budget,
            "padded": padded > 0,
            "unfilled_frame_count": max(0, frame_budget - selected_frame_count),
        }
    )


def select_kite_flow_keyframes(
    frames: list[CameraFrameRef],
    frame_budget: int,
    config: dict[str, Any] | None = None,
    selection_stats: dict[str, Any] | None = None,
    pad_to_budget: bool = False,
) -> list[CameraFrameRef]:
    """Select KITE motion keyframes, optionally padding to a fixed budget.

    Selection semantics are ported from ``m80hz/kite`` commit
    ``afead048974e5267087ef72ac2c54950995d6c5c``:
    select one coherent camera stream, sample at a fixed stride, score consecutive
    sampled frames with Farneback flow, min-max normalize the magnitudes, then greedily
    retain the strongest positions separated by more than five sampled positions. When
    ``pad_to_budget`` is true and NMS yields fewer than ``frame_budget`` frames, the
    remainder is sampled uniformly from the same camera stream. Official KITE uses
    512x512 flow inputs.
    """

    all_frames = sorted(frames, key=_frame_sort_key)
    flow_config = {}
    if config and isinstance(config.get("robofac_qa"), dict):
        flow_config = config["robofac_qa"].get("kite_flow", {}) or {}
    available_camera_names = sorted({frame.camera_name for frame in all_frames})
    camera_name = _resolve_kite_camera_name(
        available_camera_names,
        flow_config.get("camera_name"),
    )
    frames = [
        frame for frame in all_frames
        if frame.camera_name == camera_name
    ]
    frame_stride = max(1, int(flow_config.get("frame_stride", 2)))
    nms_radius = max(0, int(flow_config.get("nms_radius", 5)))
    resize_width = int(flow_config.get("resize_width", 512))
    resize_height = int(flow_config.get("resize_height", 512))
    protocol_metadata = {
        "protocol": "official_kite_single_camera",
        "camera_name": camera_name,
        "available_camera_names": available_camera_names,
        "input_frame_count": len(all_frames),
        "camera_frame_count": len(frames),
        "frame_stride": frame_stride,
        "nms_radius": nms_radius,
        "resize": [resize_width, resize_height],
    }
    if not frames:
        _record_fixed_budget_stats(
            selection_stats,
            method_frame_count=0,
            frame_budget=frame_budget,
            selected_frame_count=0,
            metadata=protocol_metadata,
        )
        return []
    cv2 = _import_cv2()

    sampled_indices = list(range(0, len(frames), frame_stride))
    sampled_frames = [frames[index] for index in sampled_indices]

    flow_scores: list[float] = []
    scored_frames: list[CameraFrameRef] = []
    previous_gray = None
    for frame_ref in sampled_frames:
        image = cv2.imread(str(Path(frame_ref.image_path)))
        if image is None:
            continue
        image = cv2.resize(image, (resize_width, resize_height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_scores.append(float(magnitude.mean()))
            scored_frames.append(frame_ref)
        previous_gray = gray

    if not flow_scores:
        selected_frames = [frames[len(frames) // 2]]
    else:
        min_score = min(flow_scores)
        max_score = max(flow_scores)
        denominator = max_score - min_score + 1e-6
        normalized_scores = [(score - min_score) / denominator for score in flow_scores]

        selected_sample_positions = _top_peaks(
            normalized_scores,
            frame_budget,
            suppression_radius=nms_radius,
        )
        selected_frames = [
            scored_frames[position] for position in selected_sample_positions
        ]

    method_frame_count = len(selected_frames)
    method_frame_keys = [_frame_identity(frame) for frame in selected_frames]
    if pad_to_budget:
        selected_frames = _uniformly_pad_keyframes(
            selected_frames,
            frames,
            frame_budget,
        )
    _record_fixed_budget_stats(
        selection_stats,
        method_frame_count=method_frame_count,
        frame_budget=frame_budget,
        selected_frame_count=len(selected_frames),
        method_frame_keys=method_frame_keys,
        metadata=protocol_metadata,
    )
    return selected_frames


def select_flow_keyframes(frames: list[CameraFrameRef], frame_budget: int) -> list[CameraFrameRef]:
    """Backward-compatible alias for official KITE motion keyframe selection."""

    return select_kite_flow_keyframes(frames, frame_budget, config={})


def _top_peaks(scores: list[float], count: int, suppression_radius: int) -> list[int]:
    candidates = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for index in candidates:
        if any(abs(index - chosen) <= suppression_radius for chosen in selected):
            continue
        selected.append(index)
        if len(selected) >= count:
            break
    return selected


def _uniformly_pad_keyframes(
    selected_frames: list[CameraFrameRef],
    all_frames: list[CameraFrameRef],
    frame_budget: int,
) -> list[CameraFrameRef]:
    """Fill a short selection uniformly over the entire remaining trajectory."""

    chosen = list(selected_frames[:frame_budget])
    if len(chosen) >= frame_budget:
        return sorted(chosen, key=_frame_sort_key)
    seen = {_frame_identity(frame) for frame in chosen}
    remaining_frames = [
        frame for frame in sorted(all_frames, key=_frame_sort_key)
        if _frame_identity(frame) not in seen
    ]
    remaining_budget = frame_budget - len(chosen)
    chosen.extend(select_uniform_keyframes(remaining_frames, remaining_budget))
    return sorted(chosen[:frame_budget], key=_frame_sort_key)


def _record_fixed_budget_stats(
    selection_stats: dict[str, Any] | None,
    *,
    method_frame_count: int,
    frame_budget: int,
    selected_frame_count: int,
    method_frame_keys: list[tuple[str, int | None, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if selection_stats is None:
        return
    padded = max(0, selected_frame_count - method_frame_count)
    selection_stats.update(
        {
            "method_frame_count": method_frame_count,
            "uniform_padded_frame_count": padded,
            "frame_budget": frame_budget,
            "selected_frame_count": selected_frame_count,
            "padded": padded > 0,
            "unfilled_frame_count": max(0, frame_budget - selected_frame_count),
            "method_frame_keys": method_frame_keys or [],
            **(metadata or {}),
        }
    )


def _resolve_kite_camera_name(
    available_camera_names: list[str],
    configured_camera_name: object,
) -> str | None:
    """Resolve one coherent camera stream for the official KITE baseline."""

    if isinstance(configured_camera_name, str) and configured_camera_name:
        return configured_camera_name
    preferred_names = ("center", "centre", "top", "hama1", "head", "front")
    for camera_name in preferred_names:
        if camera_name in available_camera_names:
            return camera_name
    return available_camera_names[0] if available_camera_names else None


def _representative_frame_for_proposal(
    frames: list[CameraFrameRef],
    time_window: Any,
    proposal: Any | None = None,
    bundle_frames: list[CameraFrameRef] | None = None,
) -> CameraFrameRef | None:
    """Return the frame closest to the proposal peak.

    ``frames`` are the proposal's candidate frames, which the selector thins to
    ``max_frames_per_camera`` and may therefore no longer contain the frame sitting on
    the peak. Candidates still decide which camera wins; when ``bundle_frames`` is given
    the timestamp is then refined against every frame of that camera so the returned
    frame lands on the peak rather than up to one stride away from it.
    """

    if not frames:
        return None
    target_sec = _proposal_peak_sec(proposal)
    if target_sec is None:
        target_sec = time_window.center_sec if time_window is not None else frames[len(frames) // 2].timestamp_sec

    def distance_to_target(frame: CameraFrameRef) -> tuple[float, int]:
        return (
            abs(frame.timestamp_sec - target_sec),
            frame.frame_index if frame.frame_index is not None else 10**12,
        )

    nearest_candidate = min(frames, key=distance_to_target)
    if not bundle_frames:
        return nearest_candidate
    same_camera_frames = [
        frame for frame in bundle_frames if frame.camera_name == nearest_candidate.camera_name
    ]
    if not same_camera_frames:
        return nearest_candidate
    return min(same_camera_frames, key=distance_to_target)


def _proposal_peak_sec(proposal: Any | None) -> float | None:
    """Return the proposal-local maximum timestamp, if the proposal carries one."""

    if proposal is None:
        return None
    metadata = getattr(proposal, "metadata", None)
    if isinstance(metadata, dict):
        peak_sec = _as_float_or_none(metadata.get("peak_sec"))
        if peak_sec is not None:
            return peak_sec
    metric_events = getattr(proposal, "metric_events", None)
    if isinstance(metric_events, list):
        for event in metric_events:
            value = getattr(event, "value", None)
            if not isinstance(value, dict):
                continue
            peak_sec = _as_float_or_none(value.get("peak_sec"))
            if peak_sec is not None:
                return peak_sec
    return None


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _frame_sort_key(frame: CameraFrameRef) -> tuple[float, int]:
    return (frame.timestamp_sec, frame.frame_index if frame.frame_index is not None else -1)


def _frame_identity(frame: CameraFrameRef) -> tuple[str, int | None, float]:
    return (frame.camera_name, frame.frame_index, frame.timestamp_sec)


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Flow keyframe selection requires opencv-python/cv2. "
            "Install opencv-python or omit the flow selector."
        ) from exc
    return cv2
