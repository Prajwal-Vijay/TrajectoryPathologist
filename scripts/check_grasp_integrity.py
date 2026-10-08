#!/usr/bin/env python3
"""Run the grasp-integrity metrics on a few ABC-130K episodes and save QA outputs.

This is the first check to run on real data, before using the metrics in any
evaluation. For each episode it writes:

- ``<out>/<task>/<episode>/signals.png``: the four signals with annotated
  failure segments shaded (needs matplotlib);
- ``<out>/<task>/<episode>/summary.json``: hold windows, per-window vision
  status and selection info, and drop/slip/transfer events;
- ``<out>/overlays/<camera>/*.jpg``: one initialization overlay per hold window
  (fingers blue, contact band yellow, rejected candidates red, chosen object
  green). Mark each as correct object / wrong object / no object to measure
  initialization accuracy.

Example:

    python scripts/check_grasp_integrity.py \\
        --data-root /path/to/ABC-130k_subset/data/train \\
        --output-dir grasp_integrity_check --max-episodes 10 --camera-sample-hz 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pathologist.abc130k.evaluation import discover_abc130k_episodes  # noqa: E402
from pathologist.adapters.abc130k_bundle import ABC130KBundleAdapter  # noqa: E402
from pathologist.metrics import grasp_integrity as gi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="grasp_integrity_check")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--bundle-id", action="append", default=[], help="Only these episodes.")
    parser.add_argument(
        "--camera-sample-hz",
        type=float,
        default=15.0,
        help="Wrist-frame rate to decode. Drops last 1-3 native frames, so prefer >= 15.",
    )
    parser.add_argument("--robot-downsample-hz", type=float, default=50.0)
    parser.add_argument("--frame-cache-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="JSON merged into metrics.grasp_integrity (e.g. empty_closed, backend options).",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_cache = Path(args.frame_cache_dir).expanduser().resolve() if args.frame_cache_dir else output / "frame_cache"

    user = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    raw = {**user, "enabled": True}
    raw["vision"] = {
        "cache_dir": str(output / "window_cache"),
        "debug_dir": str(output / "overlays"),
        **(user.get("vision") or {}),
    }

    records = [r for r in discover_abc130k_episodes(args.data_root) if r.failed_segments]
    if args.bundle_id:
        records = [r for r in records if r.bundle_id in set(args.bundle_id)]
    records = records[: args.max_episodes]
    adapter = ABC130KBundleAdapter()
    totals: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.task_name}/{record.bundle_id}", flush=True)
        bundle = adapter.load(
            record.episode_path,
            record.annotation_path,
            frame_cache_dir=frame_cache / record.task_name / record.bundle_id,
            camera_name=None,
            camera_names=["left_wrist", "right_wrist"],
            camera_sample_hz=args.camera_sample_hz,
            robot_downsample_hz=args.robot_downsample_hz,
        )
        analysis = gi.analyze_bundle(bundle, raw)
        episode_dir = output / record.task_name / record.bundle_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        failures = [[s.start_sec, s.end_sec, s.text] for s in record.failed_segments]
        summary = {
            "available": analysis.available,
            "reason": analysis.reason,
            "vision_available": analysis.vision_available,
            "vision_reason": analysis.vision_reason,
            "failure_segments": failures,
            **analysis.metadata,
        }
        (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        for windows in (analysis.metadata.get("hold_windows") or {}).values():
            for window in windows:
                key = str(window.get("vision_status"))
                totals[key] = totals.get(key, 0) + 1
        if analysis.available:
            _plot(analysis, failures, episode_dir / "signals.png")
    print(json.dumps({"hold_windows_by_vision_status": totals}, indent=2))
    print(f"Overlays for manual initialization QA: {output / 'overlays'}")


def _plot(analysis: gi.GraspAnalysis, failures: list[list], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure, axes = plt.subplots(len(gi.SIGNAL_NAMES), 1, figsize=(12, 8), sharex=True)
    for axis, name in zip(axes, gi.SIGNAL_NAMES):
        axis.plot(analysis.grid, analysis.signals[name], linewidth=1.0)
        axis.set_ylabel(name.replace("grasp_", ""), fontsize=8)
        for start, end, _ in failures:
            axis.axvspan(start, end, color="tab:red", alpha=0.15)
        for windows in (analysis.metadata.get("hold_windows") or {}).values():
            for window in windows:
                axis.axvspan(window["t_settle"], window["t_open"], color="tab:gray", alpha=0.08)
        for event in analysis.metadata.get("events", []):
            axis.axvline(event["time_sec"], color="black", linewidth=0.6, linestyle="--")
    axes[-1].set_xlabel("trajectory time (s); red = annotated failure, grey = hold windows, dashed = events")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    main()
