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
) -> list[CameraFrameRef]:
    """Select a fixed-budget frame set using a named strategy."""

    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive.")
    selector_name = selector_name.lower()
    if selector_name == "uniform":
        return select_uniform_keyframes(bundle.camera_frames, frame_budget)
    if selector_name in {"flow", "kite_flow"}:
        return select_kite_flow_keyframes(bundle.camera_frames, frame_budget, config=config or {})
    if selector_name == "metric":
        resolved_profile = metric_profile or MetricEngine(config=config or {}).compute(bundle)
        return select_metric_keyframes(bundle, resolved_profile, frame_budget, config=config or {})
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
) -> list[CameraFrameRef]:
    """Select frames from metric-overlap proposals without exposing metric values."""

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
    proposals = selector.select(metric_profile, trajectory_bundle=bundle)
    chosen: list[CameraFrameRef] = []
    seen: set[tuple[str, int | None, float]] = set()
    for proposal in proposals:
        frame = _representative_frame_for_proposal(
            proposal.candidate_frames,
            proposal.time_window,
            proposal=proposal,
        )
        if frame is None:
            continue
        key = (frame.camera_name, frame.frame_index, frame.timestamp_sec)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(frame)
        if len(chosen) >= frame_budget:
            return sorted(chosen, key=_frame_sort_key)
    if len(chosen) < frame_budget:
        for frame in select_uniform_keyframes(bundle.camera_frames, frame_budget):
            key = (frame.camera_name, frame.frame_index, frame.timestamp_sec)
            if key not in seen:
                chosen.append(frame)
                seen.add(key)
            if len(chosen) >= frame_budget:
                break
    return sorted(chosen[:frame_budget], key=_frame_sort_key)


def select_kite_flow_keyframes(
    frames: list[CameraFrameRef],
    frame_budget: int,
    config: dict[str, Any] | None = None,
) -> list[CameraFrameRef]:
    """Select motion-salient keyframes using KITE-style optical-flow peak picking.

    This reproduces the KITE keyframe-selection baseline shape: sample video frames
    at a fixed stride, score consecutive sampled frames with Farneback dense flow,
    normalize motion magnitudes, and apply fixed-radius non-maximum suppression
    over the sampled-frame index space.
    """

    cv2 = _import_cv2()
    flow_config = {}
    if config and isinstance(config.get("robofac_qa"), dict):
        flow_config = config["robofac_qa"].get("kite_flow", {}) or {}
    frame_stride = max(1, int(flow_config.get("frame_stride", 2)))
    nms_radius = max(0, int(flow_config.get("nms_radius", 5)))
    resize_width = int(flow_config.get("resize_width", 128))
    resize_height = int(flow_config.get("resize_height", 128))
    frames = sorted(frames, key=_frame_sort_key)
    if len(frames) <= frame_budget:
        return frames

    sampled_indices = list(range(0, len(frames), frame_stride))
    sampled_frames = [frames[index] for index in sampled_indices]
    if len(sampled_frames) <= frame_budget:
        return sampled_frames

    flow_scores: list[float] = []
    previous_gray = None
    for frame_ref in sampled_frames:
        image = cv2.imread(str(Path(frame_ref.image_path)))
        if image is None:
            if previous_gray is not None:
                flow_scores.append(0.0)
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
        previous_gray = gray

    if not flow_scores:
        return select_uniform_keyframes(frames, frame_budget)

    max_score = max(flow_scores)
    if max_score > 0:
        flow_scores = [score / max_score for score in flow_scores]

    selected_sample_positions = _top_peaks(
        flow_scores,
        frame_budget,
        suppression_radius=nms_radius,
    )
    selected_indices = [
        sampled_indices[min(position + 1, len(sampled_indices) - 1)]
        for position in selected_sample_positions
    ]
    if len(selected_indices) < frame_budget:
        for frame in select_uniform_keyframes(frames, frame_budget):
            index = frames.index(frame)
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= frame_budget:
                break
    return [frames[index] for index in sorted(selected_indices[:frame_budget])]


def select_flow_keyframes(frames: list[CameraFrameRef], frame_budget: int) -> list[CameraFrameRef]:
    """Backward-compatible alias for KITE-style flow keyframe selection."""

    return select_kite_flow_keyframes(frames, frame_budget, config={})


def _top_peaks(scores: list[float], count: int, suppression_radius: int) -> list[int]:
    candidates = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for index in candidates:
        if scores[index] <= 0:
            continue
        if any(abs(index - chosen) <= suppression_radius for chosen in selected):
            continue
        selected.append(index)
        if len(selected) >= count:
            break
    return selected


def _representative_frame_for_proposal(
    frames: list[CameraFrameRef],
    time_window: Any,
    proposal: Any | None = None,
) -> CameraFrameRef | None:
    if not frames:
        return None
    target_sec = _proposal_peak_sec(proposal)
    if target_sec is None:
        target_sec = time_window.center_sec if time_window is not None else frames[len(frames) // 2].timestamp_sec
    return min(
        frames,
        key=lambda frame: (
            abs(frame.timestamp_sec - target_sec),
            frame.frame_index if frame.frame_index is not None else 10**12,
        ),
    )


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


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Flow keyframe selection requires opencv-python/cv2. "
            "Install opencv-python or omit the flow selector."
        ) from exc
    return cv2
