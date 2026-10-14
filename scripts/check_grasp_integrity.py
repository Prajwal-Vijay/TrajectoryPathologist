#!/usr/bin/env python3
"""Run the grasp-integrity metrics on a few ABC-130K episodes and save QA outputs.

This is the first check to run on real data, before using the metrics in any
evaluation.

Step 1 calibrates each wrist camera's finger mask per aperture from many
episodes (spread across tasks, so backgrounds differ), and writes
``<out>/finger_calibration/<camera>.npz`` plus a PNG of the masks. Check that
PNG first: each panel should show only the fingers. Later runs reuse the files
unless ``--recalibrate`` is given.

Step 2, for each episode with an annotated failure, writes:

- ``<out>/<task>/<episode>/signals.png``: the four signals with annotated
  failure segments shaded (needs matplotlib);
- ``<out>/<task>/<episode>/summary.json``: hold windows, per-window vision
  status and selection info, and drop/slip/transfer events;
- ``<out>/overlays/<camera>/*.jpg``: one initialization overlay per hold window.
  Left: settle frame with fingers (blue), contact band (yellow), candidates
  that touched the band but were not chosen (red), the chosen object (green),
  and each candidate's rigidity. Right: a later frame with the chosen object's
  outline; the object should still fill it. Mark each as correct object / wrong
  object / no object to measure initialization accuracy.

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
from pathologist.metrics.wrist_finger_calibration import FingerCalibrator  # noqa: E402


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
    parser.add_argument("--calib-episodes", type=int, default=60, help="Episodes used to calibrate the fingers.")
    parser.add_argument("--calib-sample-hz", type=float, default=2.0)
    parser.add_argument("--calib-min-episodes", type=int, default=8, help="Minimum episodes per aperture bin.")
    parser.add_argument("--calib-bin-width", type=float, default=0.1)
    parser.add_argument("--recalibrate", action="store_true")
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

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_dir():
        sys.exit(f"--data-root does not exist: {data_root}")
    all_records = discover_abc130k_episodes(data_root)
    if not all_records:
        sys.exit(
            f"No episodes under {data_root}: expected <task>/<episode>/episode.mcap "
            "with annotation.mcap next to it."
        )
    print(
        f"Found {len(all_records)} episodes "
        f"({sum(1 for r in all_records if r.failed_segments)} with an annotated failure)."
    )
    adapter = ABC130KBundleAdapter()
    if not (raw.get("vision") or {}).get("finger_calibration") and not (raw.get("vision") or {}).get("finger_masks"):
        raw["vision"]["finger_calibration"] = _calibrate(
            all_records, adapter, raw, output / "finger_calibration", frame_cache, args
        )
        if not raw["vision"]["finger_calibration"]:
            sys.exit("Finger calibration failed for every wrist camera; see the messages above.")

    records = [r for r in all_records if r.failed_segments]
    if args.bundle_id:
        records = [r for r in records if r.bundle_id in set(args.bundle_id)]
    records = records[: args.max_episodes]
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
                confidence = (window.get("vision_info") or {}).get("confidence")
                key = f"{window.get('vision_status')}/{confidence}"
                totals[key] = totals.get(key, 0) + 1
        if analysis.available:
            _plot(analysis, failures, episode_dir / "signals.png")
    print(json.dumps({"hold_windows_by_status_and_confidence": totals}, indent=2))
    print(f"Overlays for manual initialization QA: {output / 'overlays'}")


def _calibrate(records, adapter, raw: dict, out_dir: Path, frame_cache: Path, args) -> dict[str, str]:
    """Build (or reuse) one finger calibration per wrist camera; returns camera -> file."""

    import numpy as np

    config = gi.resolve_config(raw)
    arms = config["arms"]
    vision = config["vision"]
    cameras = {arm: str(spec["camera"]) for arm, spec in arms.items()}
    paths = {camera: out_dir / f"{camera}.npz" for camera in cameras.values()}
    if not args.recalibrate and all(path.is_file() for path in paths.values()):
        print(f"Reusing finger calibration in {out_dir}")
        return {camera: str(path) for camera, path in paths.items()}

    # Spread over the (task-sorted) list so backgrounds differ between episodes.
    count = min(args.calib_episodes, len(records))
    picks = sorted({int(i) for i in np.linspace(0, len(records) - 1, num=count).round()}) if count else []
    calibrators = {
        camera: FingerCalibrator(
            work_width=int(vision["work_width"]),
            bin_width=args.calib_bin_width,
            entry_edge=str(vision["finger_entry_edge"]),
            edge_fraction=float(vision["finger_edge_fraction"]),
        )
        for camera in cameras.values()
    }
    for n, index in enumerate(picks, start=1):
        record = records[index]
        print(f"[calibration {n}/{len(picks)}] {record.task_name}/{record.bundle_id}", flush=True)
        try:
            bundle = adapter.load(
                record.episode_path,
                record.annotation_path,
                frame_cache_dir=frame_cache / "calibration" / record.task_name / record.bundle_id,
                camera_name=None,
                camera_names=sorted(set(cameras.values())),
                camera_sample_hz=args.calib_sample_hz,
                robot_downsample_hz=args.robot_downsample_hz,
            )
        except Exception as exc:  # noqa: BLE001 - one bad episode should not stop calibration
            print(f"  skipped: {type(exc).__name__}: {exc}")
            continue
        for arm, camera in cameras.items():
            inputs = gi.calibration_inputs(bundle, raw, arm)
            if inputs is None:
                continue
            frame_paths, frame_times, aperture_t, apertures, exclude = inputs
            calibrators[camera].add_episode(frame_paths, frame_times, aperture_t, apertures, exclude)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for camera, calibrator in calibrators.items():
        print(f"{camera}: frames/episodes per aperture bin: {calibrator.bin_summary()}")
        try:
            calibration = calibrator.finish(camera, min_episodes=args.calib_min_episodes)
        except ValueError as exc:
            print(f"  no calibration for {camera}: {exc}")
            continue
        written[camera] = str(calibration.save(paths[camera]))
        _save_calibration_png(calibration, out_dir / f"{camera}.png")
        print(f"  bins {calibration.centers.tolist()} -> {paths[camera]}")
    return written


def _save_calibration_png(calibration, path: Path) -> None:
    import cv2
    import numpy as np

    panels = []
    for center, mask, episodes in zip(calibration.centers, calibration.masks, calibration.episode_counts):
        panel = np.where(mask, 255, 40).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, f"a={center:.2f} n={episodes}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        panels.append(panel)
    cv2.imwrite(str(path), np.hstack(panels))


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
