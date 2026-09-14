"""Streamlit dashboard for REASSEMBLE evaluation outputs."""

from __future__ import annotations

import argparse
import bisect
import html
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathologist.reassemble.viewer import (
    CAMERA_ORDER,
    SELECTOR_ORDER,
    aggregate_score_at_time,
    build_camera_frame_index,
    flatten_peaks,
    infer_full_timeline_offset_sec,
    load_ecod_aggregate,
    load_gripper_transition_overlay,
    load_reassemble_telemetry_series_from_h5,
    load_viewer_dataset,
    nearest_frame,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect REASSEMBLE evaluation outputs.")
    parser.add_argument(
        "--output-dir",
        default="REASSEMBLE__outputs_persistence",
        help="Directory containing reassemble_eval.jsonl and reassemble_manifest.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install it in this environment with `pip install streamlit`."
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()

    st.set_page_config(
        page_title="REASSEMBLE Rapid Review",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_rapid_review_styles(st)

    try:
        dataset = load_viewer_dataset(output_dir)
    except Exception as exc:
        st.error(f"Could not load REASSEMBLE outputs: {type(exc).__name__}: {exc}")
        return

    if not dataset.bundles:
        st.warning("No included REASSEMBLE bundles were found in this output directory.")
        return

    _render_app_header(st, dataset.bundles, output_dir)
    main_panel, review_panel = st.columns([1.38, 0.92], gap="small")

    with review_panel:
        filtered_bundles = _batch_filters(st, dataset.bundles)
        if not filtered_bundles:
            st.warning("No trajectories match the current filters.")
            return
        selected_bundle = _batch_bundle_picker(st, filtered_bundles)

        aggregate = load_ecod_aggregate(output_dir, selected_bundle.bundle_id)
        frame_index = build_camera_frame_index(output_dir, selected_bundle.source_file_bundle_id)
        offset = infer_full_timeline_offset_sec(selected_bundle)

        with st.popover("Review tools", use_container_width=True):
            visible_selectors = st.multiselect(
                "Selectors",
                list(SELECTOR_ORDER),
                default=[
                    selector
                    for selector in SELECTOR_ORDER
                    if selector in selected_bundle.selectors
                ],
                format_func=_selector_display_name,
            )
            gripper_overlay = _load_gripper_overlay_for_view(st, selected_bundle, offset)

        gripper_marks = gripper_overlay.marks if gripper_overlay is not None else []
        peaks = _chronological_peaks(selected_bundle, visible_selectors)
        selected_peak = _render_review_queue(st, selected_bundle, peaks, aggregate)

    with main_panel:
        _render_query_panel(st, selected_bundle, selected_peak)
        _render_preview_video_panel(
            st,
            output_dir,
            selected_bundle,
            frame_index,
            offset,
            aggregate,
            visible_selectors,
            gripper_marks,
            focus_sec=(
                float(selected_peak.get("peak_sec") or 0.0)
                if selected_peak is not None
                else None
            ),
        )

        if selected_peak is not None:
            peak_sec = float(selected_peak.get("peak_sec") or 0.0)
            target_sec = peak_sec + offset if offset is not None else peak_sec
            st.markdown(
                (
                    '<div class="rr-section-title">'
                    f'Synchronized camera evidence · {html.escape(_format_timestamp(peak_sec))}'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if offset is None:
                st.caption("Timeline offset unavailable; matching frames with relative peak time.")
            _render_camera_panel(st, frame_index, target_sec)

        timeline_tab, telemetry_tab, details_tab = st.tabs(
            ["Signal timeline", "Raw telemetry", "Trajectory details"]
        )
        with timeline_tab:
            _render_chart(st, selected_bundle, aggregate, visible_selectors, gripper_marks)
        with telemetry_tab:
            _render_telemetry_panel(st, selected_bundle, offset)
        with details_tab:
            _render_bundle_details(st, selected_bundle)


def _inject_rapid_review_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root {
            --rr-ink: #172033;
            --rr-muted: #697386;
            --rr-border: #e7e3da;
            --rr-paper: #fffdfa;
            --rr-canvas: #f6f4ef;
            --rr-navy: #1f2a52;
        }
        html, body, [class*="css"] { font-family: Inter, sans-serif; }
        .stApp { background: var(--rr-canvas); color: var(--rr-ink); }
        .block-container {
            max-width: 100%;
            padding: .65rem .9rem 1.5rem;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none; }
        [data-testid="stSidebarCollapsedControl"] { display: none; }
        [data-testid="stHorizontalBlock"] { align-items: flex-start; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rr-border);
            border-radius: 12px;
            background: var(--rr-paper);
        }
        .rr-topbar {
            min-height: 48px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 4px 10px;
            border-bottom: 1px solid var(--rr-border);
            margin-bottom: 8px;
        }
        .rr-brand { display: flex; align-items: center; gap: 10px; }
        .rr-mark {
            width: 28px; height: 28px; border-radius: 8px;
            display: grid; place-items: center;
            background: var(--rr-navy); color: white;
            font-weight: 700; font-size: 14px;
        }
        .rr-brand-name { font-size: 14px; font-weight: 650; letter-spacing: -.01em; }
        .rr-breadcrumb { color: var(--rr-muted); font-size: 12px; margin-left: 8px; }
        .rr-top-stats { color: var(--rr-muted); font-size: 12px; }
        .rr-kicker {
            color: var(--rr-muted); font-size: 10px; font-weight: 700;
            letter-spacing: .09em; text-transform: uppercase; margin-bottom: 5px;
        }
        .rr-query {
            background: var(--rr-paper); border: 1px solid var(--rr-border);
            border-radius: 12px; padding: 13px 15px; margin: 0 0 8px;
        }
        .rr-query-title { font-size: 17px; font-weight: 650; letter-spacing: -.02em; }
        .rr-query-meta { color: var(--rr-muted); font-size: 11px; margin-top: 5px; }
        .rr-panel-heading {
            display: flex; justify-content: space-between; align-items: baseline;
            margin: 2px 0 8px;
        }
        .rr-panel-title { font-size: 15px; font-weight: 650; }
        .rr-count {
            display: inline-flex; min-width: 22px; justify-content: center;
            background: var(--rr-navy); color: white; border-radius: 999px;
            padding: 2px 7px; font-size: 10px; margin-left: 6px;
        }
        .rr-panel-meta { color: var(--rr-muted); font-size: 10px; }
        .rr-reasoning {
            background: #f2eee4; border: 1px solid #e3dccb;
            border-radius: 11px; padding: 11px 12px; margin: 9px 0;
            color: #615e58; font-size: 11px; line-height: 1.45;
        }
        .rr-reasoning strong { display: block; color: var(--rr-ink); margin-bottom: 4px; }
        .rr-section-title {
            font-size: 12px; font-weight: 650; margin: 12px 0 7px;
            color: var(--rr-ink);
        }
        .rr-event-head {
            display: flex; justify-content: space-between; gap: 8px;
            align-items: center; margin-bottom: 3px;
        }
        .rr-event-title { font-size: 12px; font-weight: 650; }
        .rr-event-time { color: var(--rr-muted); font-size: 10px; white-space: nowrap; }
        .rr-event-copy { color: var(--rr-muted); font-size: 10px; line-height: 1.4; }
        .rr-badge {
            display: inline-flex; align-items: center; border-radius: 5px;
            padding: 2px 6px; font-size: 9px; font-weight: 650;
            margin-right: 5px;
        }
        .rr-badge-metric { color: #7a2524; background: #fbe8e5; }
        .rr-badge-kite_flow { color: #8a4a08; background: #fff0d8; }
        .rr-badge-uniform { color: #24547b; background: #e8f2fa; }
        .rr-failure { color: #9a3b38; font-weight: 600; }
        .rr-neutral { color: #747069; }
        div[data-testid="stButton"] > button {
            border-radius: 8px; border-color: #ddd8cf;
            font-size: 11px; min-height: 32px;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--rr-navy); border-color: var(--rr-navy);
        }
        div[data-testid="stVideo"] { border-radius: 10px; overflow: hidden; background: #111; }
        div[data-testid="stImage"] img { border-radius: 8px; }
        div[data-testid="stExpander"] {
            border-color: var(--rr-border); border-radius: 10px; background: var(--rr-paper);
        }
        [data-baseweb="tab-list"] { gap: 4px; }
        [data-baseweb="tab"] { font-size: 11px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header(st: Any, bundles: list[Any], output_dir: Path) -> None:
    source_count = len({bundle.h5_path for bundle in bundles if bundle.h5_path})
    st.markdown(
        f"""
        <div class="rr-topbar">
          <div class="rr-brand">
            <div class="rr-mark">P</div>
            <div>
              <span class="rr-brand-name">PRISM Rapid Review</span>
              <span class="rr-breadcrumb">REASSEMBLE / {html.escape(output_dir.name)}</span>
            </div>
          </div>
          <div class="rr-top-stats">{len(bundles)} trajectories · {source_count} source files</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _batch_filters(st: Any, bundles: list[Any]) -> list[Any]:
    heading, filter_column = st.columns([3.1, 1.0], gap="small")
    with heading:
        st.markdown(
            (
                '<div class="rr-panel-heading"><div>'
                '<span class="rr-panel-title">Trajectories</span>'
                f'<span class="rr-count">{len(bundles)}</span>'
                '</div><span class="rr-panel-meta">batch review</span></div>'
            ),
            unsafe_allow_html=True,
        )
    with filter_column:
        with st.popover("Filter", use_container_width=True):
            task_kinds = sorted(
                {bundle.high_level_task_kind for bundle in bundles if bundle.high_level_task_kind}
            )
            selected_task_kinds = st.multiselect(
                "Task kind",
                task_kinds,
                default=task_kinds,
                key="rr_task_kinds",
            )
            max_coverage_pct = st.slider(
                "Max failed coverage (%)",
                0,
                100,
                100,
                key="rr_max_coverage",
            )
            search_text = st.text_input(
                "Search",
                value="",
                placeholder="Trajectory, task, or source…",
                key="rr_bundle_search",
            )

    search_lower = search_text.strip().lower()
    filtered = []
    for bundle in bundles:
        if selected_task_kinds and bundle.high_level_task_kind not in selected_task_kinds:
            continue
        coverage_pct = 100.0 * float(bundle.failed_low_level_coverage_fraction or 0.0)
        if coverage_pct > max_coverage_pct:
            continue
        searchable = " ".join(
            part
            for part in [
                bundle.bundle_id,
                bundle.high_level_task_kind or "",
                bundle.high_level_task_text or "",
                bundle.h5_path or "",
            ]
            if part
        ).lower()
        if search_lower and search_lower not in searchable:
            continue
        filtered.append(bundle)
    return filtered


def _batch_bundle_picker(st: Any, bundles: list[Any]) -> Any:
    bundle_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    bundle_id = st.selectbox(
        "Trajectory",
        list(bundle_by_id),
        format_func=lambda value: _bundle_label(bundle_by_id[value]),
        label_visibility="collapsed",
        key="rr_selected_bundle",
    )
    return bundle_by_id[bundle_id]


def _chronological_peaks(bundle: Any, visible_selectors: list[str]) -> list[dict[str, Any]]:
    selector_order = {selector: index for index, selector in enumerate(SELECTOR_ORDER)}
    return sorted(
        flatten_peaks(bundle, visible_selectors),
        key=lambda peak: (
            float(peak.get("peak_sec") or 0.0),
            selector_order.get(str(peak.get("selector_name")), len(selector_order)),
            int(peak.get("timeline_rank") or 0),
        ),
    )


def _render_review_queue(
    st: Any,
    bundle: Any,
    peaks: list[dict[str, Any]],
    aggregate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    st.markdown(
        """
        <div class="rr-reasoning">
          <strong>Selection reasoning</strong>
          PRISM surfaces telemetry-surprise events for rapid inspection alongside uniform and
          motion-only KITE baselines. Candidates below are ordered by trajectory time.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not peaks:
        st.info("Choose at least one available selector to review its candidate events.")
        return None

    selection_key = f"rr_peak_{_safe_file_stem(bundle.bundle_id)}"
    peak_by_id = {_peak_identity(peak): peak for peak in peaks}
    if st.session_state.get(selection_key) not in peak_by_id:
        st.session_state[selection_key] = next(iter(peak_by_id))

    st.markdown(
        (
            '<div class="rr-panel-heading"><div>'
            f'<span class="rr-panel-title">Events</span><span class="rr-count">{len(peaks)}</span>'
            '</div><span class="rr-panel-meta">chronological</span></div>'
        ),
        unsafe_allow_html=True,
    )
    with st.container(height=760, border=False):
        for index, peak in enumerate(peaks, start=1):
            peak_id = _peak_identity(peak)
            selected = st.session_state.get(selection_key) == peak_id
            selector = str(peak.get("selector_name") or "")
            peak_sec = float(peak.get("peak_sec") or 0.0)
            context, inside_failure = _peak_segment_context(bundle, peak_sec)
            score = aggregate_score_at_time(aggregate, peak_sec)
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="rr-event-head">
                      <span class="rr-event-title">Event {index}/{len(peaks)}</span>
                      <span class="rr-event-time">{html.escape(_format_timestamp(peak_sec))}</span>
                    </div>
                    <div>
                      <span class="rr-badge rr-badge-{html.escape(selector)}">
                        {html.escape(_selector_display_name(selector))}
                      </span>
                      <span class="{'rr-failure' if inside_failure else 'rr-neutral'}">
                        {html.escape(context)}
                      </span>
                    </div>
                    <div class="rr-event-copy">
                      Rank {html.escape(str(peak.get("timeline_rank") or "—"))}
                      {' · aggregate score ' + f'{score:.3f}' if score is not None else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Selected" if selected else "Inspect",
                    key=f"inspect_{_safe_file_stem(bundle.bundle_id)}_{index}_{peak_id}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[selection_key] = peak_id

    selected_id = st.session_state.get(selection_key)
    return peak_by_id.get(selected_id, peaks[0])


def _render_query_panel(st: Any, bundle: Any, selected_peak: dict[str, Any] | None) -> None:
    task = bundle.high_level_task_text or bundle.high_level_task_kind or "REASSEMBLE trajectory"
    meta_parts = [
        bundle.bundle_id,
        f"{float(bundle.duration_sec or 0.0):.2f}s",
    ]
    if selected_peak is not None:
        meta_parts.append(
            f"focused on {_selector_display_name(str(selected_peak.get('selector_name') or ''))} "
            f"at {_format_timestamp(float(selected_peak.get('peak_sec') or 0.0))}"
        )
    st.markdown(
        f"""
        <div class="rr-query">
          <div class="rr-kicker">Trajectory query</div>
          <div class="rr-query-title">{html.escape(str(task))}</div>
          <div class="rr-query-meta">{html.escape(' · '.join(meta_parts))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _peak_identity(peak: dict[str, Any]) -> str:
    return (
        f"{peak.get('selector_name') or 'unknown'}:"
        f"{peak.get('timeline_rank') or 0}:"
        f"{float(peak.get('peak_sec') or 0.0):.6f}"
    )


def _peak_segment_context(bundle: Any, peak_sec: float) -> tuple[str, bool]:
    for segment in bundle.failed_segments.get("low", []):
        start_sec = _float_or_none(segment.get("start_sec"))
        end_sec = _float_or_none(segment.get("end_sec"))
        if start_sec is not None and end_sec is not None and start_sec <= peak_sec <= end_sec:
            return str(segment.get("text") or "Inside failed segment"), True
    return "Candidate key event", False


def _selector_display_name(selector: str) -> str:
    return {
        "metric": "PRISM",
        "kite_flow": "KITE",
        "uniform": "Uniform",
    }.get(selector, selector.replace("_", " ").title())


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def _sidebar_bundle_picker(st: Any, bundles: list[Any]) -> Any | None:
    st.sidebar.header("Filters")
    task_kinds = sorted({bundle.high_level_task_kind for bundle in bundles if bundle.high_level_task_kind})
    selected_task_kinds = st.sidebar.multiselect("Task kind", task_kinds, default=task_kinds)
    max_coverage_pct = st.sidebar.slider("Max failed low-level coverage (%)", 0, 100, 100)
    search_text = st.sidebar.text_input("Bundle search", value="")
    h5_values = sorted({bundle.h5_path for bundle in bundles if bundle.h5_path})
    selected_h5_values = st.sidebar.multiselect("Source H5", h5_values, default=h5_values)

    filtered = []
    search_lower = search_text.strip().lower()
    for bundle in bundles:
        if selected_task_kinds and bundle.high_level_task_kind not in selected_task_kinds:
            continue
        if selected_h5_values and bundle.h5_path not in selected_h5_values:
            continue
        coverage_pct = 100.0 * float(bundle.failed_low_level_coverage_fraction or 0.0)
        if coverage_pct > max_coverage_pct:
            continue
        searchable = " ".join(
            part
            for part in [
                bundle.bundle_id,
                bundle.high_level_task_kind or "",
                bundle.high_level_task_text or "",
                bundle.h5_path or "",
            ]
            if part
        ).lower()
        if search_lower and search_lower not in searchable:
            continue
        filtered.append(bundle)

    if not filtered:
        return None
    options = {
        _bundle_label(bundle): bundle
        for bundle in filtered
    }
    label = st.sidebar.selectbox("Trajectory", list(options))
    return options[label]


def _load_gripper_overlay_for_view(st: Any, bundle: Any, offset: float | None) -> Any | None:
    show_gripper = st.checkbox("Show gripper O/C marks", value=True)
    if not show_gripper:
        return None
    threshold_mode = st.radio(
        "Gripper threshold",
        ["auto", "manual"],
        horizontal=True,
        help="Auto uses the midpoint of the gripper range inside this high-level segment.",
    )
    threshold = None
    if threshold_mode == "manual":
        threshold = float(
            st.number_input(
                "Manual gripper threshold",
                value=0.5,
                format="%.6f",
            )
        )
    min_range = float(
        st.number_input(
            "Min gripper range for O/C marks",
            value=0.005,
            min_value=0.0,
            format="%.6f",
            help="Suppresses false transitions from tiny gripper-position noise.",
        )
    )
    try:
        overlay = load_gripper_transition_overlay(
            bundle,
            offset_sec=offset,
            threshold=threshold,
            min_range=min_range,
        )
    except ModuleNotFoundError as exc:
        st.info(f"Install `{exc.name}` to show gripper open/close marks on the graph.")
        return None
    except Exception as exc:
        st.warning(f"Could not load gripper marks: {type(exc).__name__}: {exc}")
        return None
    if overlay.unavailable_reason:
        st.caption(f"Gripper O/C marks unavailable: {overlay.unavailable_reason}")
        return overlay
    st.caption(
        (
            f"Gripper samples: {overlay.sample_count} | "
            f"range: {_format_float(overlay.value_min)} to {_format_float(overlay.value_max)} | "
            f"threshold: {_format_float(overlay.threshold)} | "
            f"O/C marks: {len(overlay.marks)}"
        )
    )
    if overlay.sample_count >= 2 and not overlay.marks:
        st.caption("No open/close transitions crossed the selected threshold in this segment.")
    return overlay


def _format_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _render_preview_video_panel(
    st: Any,
    output_dir: Path,
    bundle: Any,
    frame_index: dict[str, list[Any]],
    offset: float | None,
    aggregate: dict[str, Any] | None,
    visible_selectors: list[str],
    gripper_marks: list[Any],
    *,
    focus_sec: float | None = None,
) -> None:
    duration_sec = float(bundle.duration_sec or 0.0)
    if not frame_index:
        st.warning("No cached camera frames were found for this source trajectory.")
        return
    if offset is None:
        st.warning("Could not infer the full-timeline start of this high-level trajectory.")
        return
    if duration_sec <= 0:
        st.warning("This trajectory does not have a valid high-level duration.")
        return

    start_sec = float(offset)
    end_sec = start_sec + duration_sec
    available_frames = _camera_timelines_in_window(frame_index, start_sec, end_sec)
    if not available_frames:
        st.warning(
            f"No cached camera frames fall inside {start_sec:.3f}s to {end_sec:.3f}s."
        )
        return

    start_time = max(0, int((focus_sec or 0.0) - 1.0))
    safe_id = _safe_file_stem(bundle.bundle_id)
    fps_key = f"{safe_id}_preview_fps"
    if fps_key not in st.session_state:
        st.session_state[fps_key] = 15
    fps = max(2, int(st.session_state[fps_key]))
    video_path = output_dir / "viewer_cache" / f"{safe_id}_fps{fps}.webm"
    if video_path.is_file():
        st.video(str(video_path), format="video/webm", start_time=start_time)
        st.caption(
            f"Synchronized review playback · {duration_sec:.2f}s · {fps} FPS"
            + (
                f" · opened near {_format_timestamp(float(focus_sec))}"
                if focus_sec is not None
                else ""
            )
        )
        return

    controls = st.columns([1.0, 1.3, 2.3], gap="small")
    with controls[0]:
        st.slider("Preview FPS", min_value=2, max_value=15, step=1, key=fps_key)
    fps = max(2, int(st.session_state[fps_key]))
    video_path = output_dir / "viewer_cache" / f"{safe_id}_fps{fps}.webm"
    estimated_frames = int(duration_sec * fps) + 1
    with controls[1]:
        generate = st.button(
            "Render preview",
            key=f"{safe_id}_generate_video",
            use_container_width=True,
        )
    with controls[2]:
        st.caption(
            f"{duration_sec:.2f}s · {estimated_frames} rendered frames · "
            f"{'cached' if video_path.is_file() else 'not rendered'}"
        )

    if estimated_frames > 2400:
        st.warning(
            "This preview will be large at the selected FPS. Lower the FPS if generation is slow or disk space is tight."
        )

    if generate:
        with st.spinner("Rendering preview video from cached frames..."):
            try:
                _write_preview_video(
                    video_path=video_path,
                    bundle=bundle,
                    frame_index=available_frames,
                    offset_sec=start_sec,
                    duration_sec=duration_sec,
                    fps=fps,
                    aggregate=aggregate,
                    visible_selectors=visible_selectors,
                    gripper_marks=gripper_marks,
                )
            except ModuleNotFoundError as exc:
                st.error(f"Missing dependency for video rendering: {exc.name}. Install `opencv-python`.")
            except Exception as exc:
                st.error(f"Could not render preview video: {type(exc).__name__}: {exc}")

    if not video_path.is_file():
        st.info("Render this trajectory once to enable synchronized local playback.")
        return
    st.video(str(video_path), format="video/webm", start_time=start_time)


def _write_preview_video(
    *,
    video_path: Path,
    bundle: Any,
    frame_index: dict[str, list[Any]],
    offset_sec: float,
    duration_sec: float,
    fps: int,
    aggregate: dict[str, Any] | None,
    visible_selectors: list[str],
    gripper_marks: list[Any],
) -> None:
    import cv2
    import numpy as np

    video_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1440
    height = 900
    chart_height = 320
    camera_top = chart_height + 24
    camera_height = height - camera_top
    fps = max(1, int(fps))
    total_frames = max(1, int(duration_sec * fps) + 1)
    writer_path = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    writer = _open_browser_video_writer(cv2, writer_path, fps, width, height)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a browser-compatible WebM writer.")

    camera_names = _ordered_camera_names(frame_index)
    timelines = {
        camera_name: (
            [float(frame.timestamp_sec) for frame in frames],
            frames,
        )
        for camera_name, frames in frame_index.items()
    }
    try:
        for frame_number in range(total_frames):
            relative_sec = min(duration_sec, frame_number / float(fps))
            target_sec = offset_sec + relative_sec
            canvas = np.full((height, width, 3), 246, dtype=np.uint8)
            _draw_preview_chart(
                cv2=cv2,
                np=np,
                canvas=canvas,
                bundle=bundle,
                aggregate=aggregate,
                visible_selectors=visible_selectors,
                gripper_marks=gripper_marks,
                duration_sec=duration_sec,
                playhead_sec=relative_sec,
                chart_height=chart_height,
            )
            _draw_preview_cameras(
                cv2=cv2,
                np=np,
                canvas=canvas,
                camera_names=camera_names,
                timelines=timelines,
                target_sec=target_sec,
                top=camera_top,
                height=camera_height,
            )
            writer.write(canvas)
    finally:
        writer.release()
    writer_path.replace(video_path)


def _open_browser_video_writer(cv2: Any, path: Path, fps: int, width: int, height: int) -> Any:
    for fourcc_name in ("VP80", "VP90"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    return cv2.VideoWriter()


def _draw_preview_chart(
    *,
    cv2: Any,
    np: Any,
    canvas: Any,
    bundle: Any,
    aggregate: dict[str, Any] | None,
    visible_selectors: list[str],
    gripper_marks: list[Any],
    duration_sec: float,
    playhead_sec: float,
    chart_height: int,
) -> None:
    left = 70
    right = canvas.shape[1] - 28
    top = 32
    bottom = chart_height - 46
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], chart_height), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "Aggregate ECOD score with selected peaks",
        (left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    timestamps, raw_scores, smoothed_scores = _aggregate_series(aggregate)
    values = raw_scores + smoothed_scores
    for selector in visible_selectors:
        for peak in bundle.peaks_by_selector.get(selector, []):
            score = aggregate_score_at_time(aggregate, float(peak.get("peak_sec") or 0.0))
            if score is not None:
                values.append(float(score))
    gripper_marker_score = _gripper_marker_y_value(values)
    if gripper_marks:
        values.append(gripper_marker_score)
    y_min = min(0.0, min(values)) if values else 0.0
    y_max = max(values) if values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_at(timestamp_sec: float) -> int:
        if duration_sec <= 0:
            return left
        return int(round(left + (max(0.0, min(duration_sec, timestamp_sec)) / duration_sec) * plot_width))

    def y_at(score: float) -> int:
        return int(round(bottom - ((score - y_min) / (y_max - y_min)) * plot_height))

    for segment in bundle.failed_segments.get("low", []):
        start_sec = _float_or_none(segment.get("start_sec"))
        end_sec = _float_or_none(segment.get("end_sec"))
        if start_sec is None or end_sec is None:
            continue
        x0 = x_at(start_sec)
        x1 = x_at(end_sec)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, top), (x1, bottom), (220, 225, 255), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    cv2.rectangle(canvas, (left, top), (right, bottom), (210, 210, 210), 1)
    cv2.line(canvas, (left, bottom), (right, bottom), (60, 60, 60), 1)
    cv2.line(canvas, (left, top), (left, bottom), (60, 60, 60), 1)
    cv2.putText(canvas, "0s", (left, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{duration_sec:.1f}s",
        (right - 78, bottom + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    _draw_series(cv2, np, canvas, timestamps, raw_scores, x_at, y_at, (170, 170, 170), 1)
    _draw_series(cv2, np, canvas, timestamps, smoothed_scores, x_at, y_at, (25, 25, 25), 2)

    selector_colors = {"uniform": (168, 120, 76), "kite_flow": (24, 133, 245), "metric": (86, 87, 228)}
    for selector in visible_selectors:
        color = selector_colors.get(selector, (90, 90, 90))
        for peak in bundle.peaks_by_selector.get(selector, []):
            peak_sec = float(peak.get("peak_sec") or 0.0)
            score = aggregate_score_at_time(aggregate, peak_sec)
            y_value = y_at(float(score if score is not None else y_min))
            x_value = x_at(peak_sec)
            cv2.circle(canvas, (x_value, y_value), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(peak.get("timeline_rank") or ""),
                (x_value + 8, max(top + 14, y_value - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    for mark in gripper_marks:
        mark_sec = float(mark.timestamp_sec)
        x_value = x_at(mark_sec)
        y_value = y_at(gripper_marker_score)
        _draw_dashed_vertical_line(
            cv2=cv2,
            canvas=canvas,
            x=x_value,
            y0=top,
            y1=bottom,
            color=(235, 122, 40),
            thickness=2,
        )
        cv2.putText(
            canvas,
            str(mark.label),
            (x_value - 5, max(top + 14, y_value - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 122, 40),
            2,
            cv2.LINE_AA,
        )

    playhead_x = x_at(playhead_sec)
    cv2.line(canvas, (playhead_x, top), (playhead_x, bottom), (0, 0, 0), 2)
    cv2.putText(
        canvas,
        f"{playhead_sec:.2f}s",
        (min(playhead_x + 8, right - 80), top + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_preview_cameras(
    *,
    cv2: Any,
    np: Any,
    canvas: Any,
    camera_names: list[str],
    timelines: dict[str, tuple[list[float], list[Any]]],
    target_sec: float,
    top: int,
    height: int,
) -> None:
    if not camera_names:
        cv2.putText(
            canvas,
            "No cached camera frames in this high-level interval",
            (60, top + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        return
    columns = min(3, len(camera_names))
    tile_width = canvas.shape[1] // columns
    tile_height = height
    for index, camera_name in enumerate(camera_names[:columns]):
        x0 = index * tile_width
        x1 = canvas.shape[1] if index == columns - 1 else (index + 1) * tile_width
        tile = canvas[top:top + tile_height, x0:x1]
        tile[:] = (238, 238, 238)
        frame = _nearest_frame_from_timeline(timelines[camera_name], target_sec)
        if frame is None:
            cv2.putText(tile, camera_name, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(tile, "missing", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
            continue
        image = cv2.imread(str(frame.image_path))
        if image is None:
            cv2.putText(tile, camera_name, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(tile, "could not read frame", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
            continue
        label_height = 48
        image_area = tile[label_height:tile_height - 8, 8:x1 - x0 - 8]
        _paste_fit(cv2, np, image_area, image)
        delta = frame.timestamp_sec - target_sec
        label = f"{camera_name}  t={frame.timestamp_sec:.3f}s  delta={delta:+.3f}s"
        cv2.putText(tile, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)


def _draw_series(
    cv2: Any,
    np: Any,
    canvas: Any,
    timestamps: list[float],
    scores: list[float],
    x_at: Any,
    y_at: Any,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(timestamps) != len(scores) or len(timestamps) < 2:
        return
    points = np.array([[x_at(timestamp), y_at(score)] for timestamp, score in zip(timestamps, scores)], dtype=np.int32)
    cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)


def _draw_dashed_vertical_line(
    *,
    cv2: Any,
    canvas: Any,
    x: int,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 10,
    gap_length: int = 7,
) -> None:
    y = min(y0, y1)
    end = max(y0, y1)
    while y < end:
        dash_end = min(y + dash_length, end)
        cv2.line(canvas, (x, y), (x, dash_end), color, thickness, cv2.LINE_AA)
        y = dash_end + gap_length


def _paste_fit(cv2: Any, np: Any, target: Any, image: Any) -> None:
    target_height, target_width = target.shape[:2]
    image_height, image_width = image.shape[:2]
    if target_height <= 0 or target_width <= 0 or image_height <= 0 or image_width <= 0:
        return
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    target[:] = (20, 20, 20)
    x0 = (target_width - resized_width) // 2
    y0 = (target_height - resized_height) // 2
    target[y0:y0 + resized_height, x0:x0 + resized_width] = resized


def _camera_timelines_in_window(
    frame_index: dict[str, list[Any]],
    start_sec: float,
    end_sec: float,
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for camera_name, frames in frame_index.items():
        selected = [
            frame
            for frame in frames
            if start_sec <= float(frame.timestamp_sec) <= end_sec
        ]
        if selected:
            result[camera_name] = selected
    return result


def _nearest_frame_from_timeline(timeline: tuple[list[float], list[Any]], target_sec: float) -> Any | None:
    timestamps, frames = timeline
    if not timestamps:
        return None
    index = bisect.bisect_left(timestamps, target_sec)
    if index <= 0:
        return frames[0]
    if index >= len(timestamps):
        return frames[-1]
    before = frames[index - 1]
    after = frames[index]
    if abs(float(before.timestamp_sec) - target_sec) <= abs(float(after.timestamp_sec) - target_sec):
        return before
    return after


def _aggregate_series(aggregate: dict[str, Any] | None) -> tuple[list[float], list[float], list[float]]:
    if aggregate is None:
        return [], [], []
    timestamps = _numeric_list(aggregate.get("timestamps_sec"))
    raw_scores = _numeric_list(aggregate.get("raw_scores"))
    smoothed_scores = _numeric_list(aggregate.get("smoothed_scores"))
    if raw_scores and len(raw_scores) != len(timestamps):
        raw_scores = []
    if smoothed_scores and len(smoothed_scores) != len(timestamps):
        smoothed_scores = []
    return timestamps, raw_scores, smoothed_scores


def _source_metric_series_from_aggregate(aggregate: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if aggregate is None:
        return {}
    raw_series = aggregate.get("source_metric_series")
    if not isinstance(raw_series, dict):
        return {}

    parsed: dict[str, dict[str, Any]] = {}
    for metric_name, payload in raw_series.items():
        if not isinstance(metric_name, str) or not isinstance(payload, dict):
            continue
        timestamps = _numeric_list(payload.get("timestamps_sec"))
        values = _numeric_list(payload.get("raw_values"))
        if not timestamps or len(timestamps) != len(values):
            continue
        parsed[metric_name] = {
            "timestamps_sec": timestamps,
            "raw_values": values,
            "unit": payload.get("unit") if isinstance(payload.get("unit"), str) else None,
            "description": payload.get("description") if isinstance(payload.get("description"), str) else None,
        }
    return parsed


def _plotly_gripper_marker_y(
    aggregate: dict[str, Any] | None,
    bundle: Any,
    visible_selectors: list[str],
) -> float:
    _, raw_scores, smoothed_scores = _aggregate_series(aggregate)
    values = raw_scores + smoothed_scores
    for selector in visible_selectors:
        for peak in bundle.peaks_by_selector.get(selector, []):
            score = aggregate_score_at_time(aggregate, float(peak.get("peak_sec") or 0.0))
            if score is not None:
                values.append(float(score))
    return _gripper_marker_y_value(values)


def _gripper_marker_y_value(values: list[float]) -> float:
    if not values:
        return 1.0
    y_min = min(0.0, min(values))
    y_max = max(values)
    span = y_max - y_min
    if span <= 0:
        return y_max + 1.0
    return y_max + 0.08 * span


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    numbers = [_float_or_none(item) for item in value]
    return [number for number in numbers if number is not None]


def _ordered_camera_names(frame_index: dict[str, list[Any]]) -> list[str]:
    camera_names = [camera for camera in CAMERA_ORDER if camera in frame_index]
    camera_names.extend(sorted(camera for camera in frame_index if camera not in camera_names))
    return camera_names


def _render_chart(
    st: Any,
    bundle: Any,
    aggregate: dict[str, Any] | None,
    visible_selectors: list[str],
    gripper_marks: list[Any],
) -> None:
    st.subheader("Aggregate ECOD Score")
    if aggregate is None:
        st.warning("No `ecod_aggregates` JSON was found for this trajectory. Run viewer prep with plots enabled.")
        return
    timestamps = aggregate.get("timestamps_sec") if isinstance(aggregate.get("timestamps_sec"), list) else []
    raw_scores = aggregate.get("raw_scores") if isinstance(aggregate.get("raw_scores"), list) else []
    smoothed_scores = aggregate.get("smoothed_scores") if isinstance(aggregate.get("smoothed_scores"), list) else []
    if not timestamps:
        st.warning("Aggregate JSON exists, but it does not contain timestamps.")
        return
    source_series = _source_metric_series_from_aggregate(aggregate)
    chart_mode_options = ["Aggregate ECOD"]
    if source_series:
        chart_mode_options.extend(["Raw source metrics", "Aggregate + raw source metrics"])
    chart_mode = st.radio(
        "Chart traces",
        chart_mode_options,
        horizontal=True,
        help="Raw source metrics are plotted without smoothing or uncertainty transforms.",
    )
    selected_source_metrics: list[str] = []
    if source_series and chart_mode != "Aggregate ECOD":
        selected_source_metrics = st.multiselect(
            "Raw source metrics",
            list(source_series),
            default=list(source_series),
            help="These are the raw metric streams that feed the ECOD aggregate.",
        )
    elif not source_series:
        st.caption("Raw source metrics are not available in this output. Rerun viewer prep to write them.")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        _render_chart_fallback(
            st,
            timestamps,
            raw_scores,
            smoothed_scores,
            source_series,
            selected_source_metrics,
            chart_mode,
        )
        return

    fig = go.Figure()
    show_aggregate = chart_mode in {"Aggregate ECOD", "Aggregate + raw source metrics"}
    show_source_metrics = chart_mode in {"Raw source metrics", "Aggregate + raw source metrics"}
    if show_aggregate and raw_scores:
        fig.add_trace(go.Scatter(x=timestamps, y=raw_scores, mode="lines", name="raw", line={"color": "#9a9a9a"}))
    if show_aggregate and smoothed_scores:
        fig.add_trace(
            go.Scatter(x=timestamps, y=smoothed_scores, mode="lines", name="smoothed", line={"color": "#111111"})
        )
    if show_source_metrics:
        for metric_name in selected_source_metrics:
            series = source_series.get(metric_name)
            if not series:
                continue
            fig.add_trace(
                go.Scatter(
                    x=series["timestamps_sec"],
                    y=series["raw_values"],
                    mode="lines",
                    name=f"{metric_name} raw",
                    hovertemplate=(
                        "metric=%{fullData.name}<br>"
                        "time=%{x:.3f}s<br>"
                        "value=%{y:.6g}<extra></extra>"
                    ),
                )
            )
    for segment in bundle.failed_segments.get("low", []):
        start_sec = segment.get("start_sec")
        end_sec = segment.get("end_sec")
        if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)):
            fig.add_vrect(
                x0=float(start_sec),
                x1=float(end_sec),
                fillcolor="#d62728",
                opacity=0.12,
                line_width=0,
                annotation_text=str(segment.get("text") or "failed low"),
                annotation_position="top left",
            )
    colors = {"uniform": "#4c78a8", "kite_flow": "#f58518", "metric": "#e45756"}
    for selector in visible_selectors:
        peaks = bundle.peaks_by_selector.get(selector, [])
        if show_aggregate:
            x_values = [float(peak.get("peak_sec") or 0.0) for peak in peaks]
            y_values = [
                aggregate_score_at_time(aggregate, float(peak.get("peak_sec") or 0.0)) or 0.0
                for peak in peaks
            ]
            hover = [_peak_hover_text(selector, peak) for peak in peaks]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers+text",
                    name=selector,
                    text=[str(peak.get("timeline_rank") or "") for peak in peaks],
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                    marker={"size": 11, "color": colors.get(selector, "#666666")},
                )
            )
        else:
            for peak in peaks:
                peak_sec = float(peak.get("peak_sec") or 0.0)
                fig.add_vline(
                    x=peak_sec,
                    line_color=colors.get(selector, "#666666"),
                    line_width=1,
                    line_dash="dot",
                    annotation_text=f"{selector} {peak.get('timeline_rank') or ''}",
                    annotation_position="top",
                )
    if gripper_marks:
        marker_y = _plotly_gripper_marker_y(aggregate, bundle, visible_selectors)
        for mark in gripper_marks:
            fig.add_vline(
                x=float(mark.timestamp_sec),
                line_color="#1f77b4",
                line_width=2,
                line_dash="dash",
                annotation_text=str(mark.label),
                annotation_position="top",
                annotation_font_color="#1f77b4",
            )
        fig.add_trace(
            go.Scatter(
                x=[float(mark.timestamp_sec) for mark in gripper_marks],
                y=[marker_y for _ in gripper_marks],
                mode="markers",
                name="gripper O/C",
                hovertext=[
                    (
                        f"gripper: {mark.state}<br>"
                        f"transition: {mark.previous_state} -> {mark.state}<br>"
                        f"time: {mark.timestamp_sec:.3f}s<br>"
                        f"value: {mark.raw_value:.4f}"
                    )
                    for mark in gripper_marks
                ],
                hoverinfo="text",
                marker={"size": 1, "color": "rgba(31,119,180,0.01)"},
                showlegend=True,
            )
        )
    fig.update_layout(
        xaxis_title="Seconds within high-level trajectory",
        yaxis_title=(
            "Aggregate ECOD score"
            if chart_mode == "Aggregate ECOD"
            else "Raw metric value"
            if chart_mode == "Raw source metrics"
            else "Mixed raw metric / ECOD values"
        ),
        height=430,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_chart_fallback(
    st: Any,
    timestamps: list[Any],
    raw_scores: list[Any],
    smoothed_scores: list[Any],
    source_series: dict[str, dict[str, Any]],
    selected_source_metrics: list[str],
    chart_mode: str,
) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        st.info("Install `plotly` or `pandas` for an inline line chart. Showing aggregate length only.")
        st.write({"sample_count": len(timestamps)})
        return
    data: dict[str, Any] = {"timestamp_sec": timestamps}
    if chart_mode in {"Aggregate ECOD", "Aggregate + raw source metrics"} and raw_scores:
        data["raw"] = raw_scores
    if chart_mode in {"Aggregate ECOD", "Aggregate + raw source metrics"} and smoothed_scores:
        data["smoothed"] = smoothed_scores
    if chart_mode in {"Raw source metrics", "Aggregate + raw source metrics"}:
        for metric_name in selected_source_metrics:
            series = source_series.get(metric_name)
            if not series or series["timestamps_sec"] != timestamps:
                continue
            data[f"{metric_name} raw"] = series["raw_values"]
    st.line_chart(pd.DataFrame(data), x="timestamp_sec")


def _render_telemetry_panel(st: Any, bundle: Any, offset: float | None) -> None:
    st.subheader("Raw Telemetry Streams")
    h5_path = bundle.h5_path
    duration_sec = _float_or_none(bundle.duration_sec)
    start_sec = offset
    if start_sec is None:
        start_sec = _float_or_none(bundle.manifest_entry.get("high_level_segment_start_sec_in_file"))
    if not h5_path or duration_sec is None or start_sec is None:
        st.info("Raw telemetry unavailable for this trajectory.")
        return

    controls = st.columns([1.2, 1.0], gap="small")
    with controls[0]:
        max_samples = int(
            st.slider(
                "Telemetry max samples",
                min_value=500,
                max_value=10000,
                value=3000,
                step=500,
                help="Downsamples dense H5 streams for responsive plotting.",
            )
        )
    with controls[1]:
        st.caption(
            f"High-level window: {float(start_sec):.3f}s to {float(start_sec) + duration_sec:.3f}s in source H5."
        )

    try:
        streams = _cached_telemetry_series(
            h5_path=str(h5_path),
            start_sec=float(start_sec),
            duration_sec=duration_sec,
            max_samples=max_samples,
        )
    except ModuleNotFoundError as exc:
        st.info(f"Install `{exc.name}` to show raw telemetry streams.")
        return
    except Exception as exc:
        st.warning(f"Could not load raw telemetry: {type(exc).__name__}: {exc}")
        return

    streams = [stream for stream in streams if not getattr(stream, "unavailable_reason", None)]
    if not streams:
        st.info("No raw telemetry streams were found inside this high-level interval.")
        return

    stream_by_label = {
        f"{stream.name} ({stream.sample_count} samples)": stream
        for stream in streams
    }
    selected_label = st.selectbox("Telemetry stream", list(stream_by_label))
    stream = stream_by_label[selected_label]
    component_names = list(stream.values_by_component)
    default_components = ["magnitude"] if "magnitude" in component_names else component_names[: min(3, len(component_names))]
    selected_components = st.multiselect(
        "Components",
        component_names,
        default=default_components,
        help="Use magnitude for an at-a-glance force/velocity view, or inspect individual axes/joints.",
    )
    if not selected_components:
        st.info("Select at least one telemetry component.")
        return

    _render_telemetry_chart(
        st,
        timestamps=stream.timestamps_sec,
        values_by_component=stream.values_by_component,
        selected_components=selected_components,
        stream_name=stream.name,
        source_dataset=stream.source_dataset,
    )


def _render_telemetry_chart(
    st: Any,
    *,
    timestamps: list[float],
    values_by_component: dict[str, list[float]],
    selected_components: list[str],
    stream_name: str,
    source_dataset: str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        _render_telemetry_chart_fallback(st, timestamps, values_by_component, selected_components)
        return

    fig = go.Figure()
    for component in selected_components:
        values = values_by_component.get(component)
        if not values:
            continue
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=component,
                hovertemplate=(
                    "component=%{fullData.name}<br>"
                    "time=%{x:.3f}s<br>"
                    "value=%{y:.6g}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Seconds within high-level trajectory",
        yaxis_title=stream_name,
        height=330,
        margin={"l": 40, "r": 20, "t": 24, "b": 40},
        title={"text": source_dataset, "font": {"size": 13}},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_telemetry_chart_fallback(
    st: Any,
    timestamps: list[float],
    values_by_component: dict[str, list[float]],
    selected_components: list[str],
) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        st.write({"sample_count": len(timestamps), "components": selected_components})
        return
    data: dict[str, Any] = {"timestamp_sec": timestamps}
    for component in selected_components:
        values = values_by_component.get(component)
        if values and len(values) == len(timestamps):
            data[component] = values
    st.line_chart(pd.DataFrame(data), x="timestamp_sec")


def _cached_telemetry_series(
    *,
    h5_path: str,
    start_sec: float,
    duration_sec: float,
    max_samples: int,
) -> list[Any]:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return _load_telemetry_series_uncached(
            h5_path=h5_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            max_samples=max_samples,
        )
    cached_loader = st.cache_data(show_spinner=False)(_load_telemetry_series_uncached)
    return cached_loader(
        h5_path=h5_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
        max_samples=max_samples,
    )


def _load_telemetry_series_uncached(
    *,
    h5_path: str,
    start_sec: float,
    duration_sec: float,
    max_samples: int,
) -> list[Any]:
    return load_reassemble_telemetry_series_from_h5(
        h5_path=h5_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
        max_samples=max_samples,
    )


def _render_bundle_details(st: Any, bundle: Any) -> None:
    st.subheader("Trajectory Details")
    coverage = bundle.failed_low_level_coverage_fraction
    st.metric("Duration", f"{float(bundle.duration_sec or 0.0):.2f}s")
    st.metric("Failed low-level coverage", f"{100.0 * float(coverage or 0.0):.2f}%")
    st.write(
        {
            "bundle_id": bundle.bundle_id,
            "task": bundle.high_level_task_text,
            "kind": bundle.high_level_task_kind,
            "source": Path(bundle.h5_path).name if bundle.h5_path else None,
            "source_file_bundle_id": bundle.source_file_bundle_id,
        }
    )
    low_segments = bundle.failed_segments.get("low", [])
    if low_segments:
        st.caption("Failed low-level segments")
        st.table(
            [
                {
                    "text": segment.get("text"),
                    "start_sec": segment.get("start_sec"),
                    "end_sec": segment.get("end_sec"),
                }
                for segment in low_segments
            ]
        )


def _render_peak_table(st: Any, peaks: list[dict[str, Any]]) -> None:
    rows = [_peak_table_row(peak) for peak in peaks]
    try:
        import pandas as pd
    except ModuleNotFoundError:
        st.table(rows)
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _peak_picker(st: Any, peaks: list[dict[str, Any]]) -> dict[str, Any] | None:
    options = {_peak_label(peak): peak for peak in peaks}
    label = st.selectbox("Peak to inspect", list(options))
    return options.get(label)


def _render_camera_panel(st: Any, frame_index: dict[str, list[Any]], target_sec: float) -> None:
    if not frame_index:
        st.warning("No frame cache images were found for this source trajectory.")
        return
    camera_names = [camera for camera in CAMERA_ORDER if camera in frame_index]
    camera_names.extend(sorted(camera for camera in frame_index if camera not in camera_names))
    columns = st.columns(max(1, min(3, len(camera_names))))
    for index, camera_name in enumerate(camera_names):
        match = nearest_frame(frame_index, camera_name, target_sec)
        column = columns[index % len(columns)]
        with column:
            if match is None:
                st.warning(f"No frames for {camera_name}")
                continue
            st.image(
                str(match.frame.image_path),
                caption=(
                    f"{camera_name} | frame {match.frame.frame_index} | "
                    f"t={match.frame.timestamp_sec:.3f}s | delta={match.delta_sec:+.3f}s"
                ),
                use_container_width=True,
            )


def _bundle_label(bundle: Any) -> str:
    coverage = 100.0 * float(bundle.failed_low_level_coverage_fraction or 0.0)
    return f"{bundle.bundle_id} | {bundle.high_level_task_kind or 'unknown'} | {coverage:.1f}%"


def _peak_label(peak: dict[str, Any]) -> str:
    selector = str(peak.get("selector_name"))
    rank = peak.get("timeline_rank")
    peak_sec = float(peak.get("peak_sec") or 0.0)
    return f"{selector} #{rank} at {peak_sec:.3f}s"


def _peak_table_row(peak: dict[str, Any]) -> dict[str, Any]:
    return {
        "selector": peak.get("selector_name"),
        "rank": peak.get("timeline_rank"),
        "peak_sec": _round_or_none(peak.get("peak_sec")),
        "persistence": _round_or_none(peak.get("persistence")),
        "area_above_base": _round_or_none(peak.get("area_above_base")),
        "width_sec": _round_or_none(peak.get("width_sec")),
        "matched_camera": peak.get("camera_name"),
        "frame_index": peak.get("frame_index"),
    }


def _peak_hover_text(selector: str, peak: dict[str, Any]) -> str:
    parts = [
        f"selector: {selector}",
        f"rank: {peak.get('timeline_rank')}",
        f"peak_sec: {_round_or_none(peak.get('peak_sec'))}",
        f"persistence: {_round_or_none(peak.get('persistence'))}",
        f"area: {_round_or_none(peak.get('area_above_base'))}",
        f"width: {_round_or_none(peak.get('width_sec'))}",
    ]
    return "<br>".join(parts)


def _round_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float, bool)):
        return round(float(value), 4)
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _safe_file_stem(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe or "trajectory"


if __name__ == "__main__":
    main()
