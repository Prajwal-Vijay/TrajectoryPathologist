"""CLI for deterministic REASSEMBLE failure-segment recall evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathologist.reassemble import run_reassemble_eval_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Recall@K evaluation on REASSEMBLE trajectories."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--h5-path", help="Path to one REASSEMBLE .h5 trajectory.")
    input_group.add_argument("--data-root", help="Directory containing REASSEMBLE .h5 trajectories.")
    parser.add_argument("--output-dir", default="reassemble_outputs", help="Directory for evaluation outputs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of selected peaks to evaluate.")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Selectors to compare. Defaults to all three: uniform, kite_flow, metric.",
    )
    parser.add_argument(
        "--segment-level",
        choices=["high", "low", "both"],
        default="low",
        help="Which failure segment level to score. Defaults to low-level failures inside each high-level demo.",
    )
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default=None,
        help="Rank metric proposals by area or persistence. Defaults to the config value (area in default_config.json).",
    )
    parser.add_argument(
        "--evaluation-unit",
        choices=["failed_high_level_action", "successful_high_level_action"],
        default="failed_high_level_action",
        help="Which high-level demos to include. Defaults to failed high-level actions.",
    )
    parser.add_argument("--config", dest="config_path", help="Optional pathologist config JSON override.")
    parser.add_argument(
        "--robot-downsample-hz",
        type=float,
        default=None,
        help="Downsample robot streams before metric computation. Defaults to config value, usually 50 Hz.",
    )
    parser.add_argument(
        "--camera-name",
        "--camera",
        dest="camera_names",
        action="append",
        choices=["hand", "hama1", "hama2"],
        help=(
            "RGB camera dataset to decode/use. Can be passed multiple times. "
            "Defaults to hama1, hama2, and hand."
        ),
    )
    parser.add_argument(
        "--include-event-camera",
        action="store_true",
        default=None,
        help="Also decode capture_node-camera-image.",
    )
    parser.add_argument(
        "--no-extract-frames",
        action="store_true",
        help="Skip frame extraction and visual-flow metric for faster telemetry-only debugging.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default=None,
        help="Optional shared decoded-frame cache. Defaults to <output-dir>/frame_cache.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write raw/smoothed ECOD aggregate plots and local-maxima JSON under ecod_aggregates/.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for folder evaluation.")
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Optional cap on included high-level demos across all input files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest/eval outputs in --output-dir instead of truncating them.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_reassemble_eval_pipeline(
        h5_path=Path(args.h5_path) if args.h5_path is not None else None,
        data_root=Path(args.data_root) if args.data_root is not None else None,
        output_dir=args.output_dir,
        top_k=args.top_k,
        segment_level=args.segment_level,
        config_path=args.config_path,
        robot_downsample_hz=args.robot_downsample_hz,
        camera_names=args.camera_names,
        include_event_camera=args.include_event_camera,
        extract_frames=False if args.no_extract_frames else None,
        frame_cache_dir=args.frame_cache_dir,
        max_files=args.max_files,
        selectors=args.selectors,
        metric_ranking=args.metric_ranking,
        evaluation_unit=args.evaluation_unit,
        max_demos=args.max_demos,
        plot_ecod_aggregates=args.plot,
        resume=args.resume,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
