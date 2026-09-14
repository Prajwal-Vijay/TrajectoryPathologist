"""Prepare REASSEMBLE evaluation outputs for the local viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathologist.reassemble import run_reassemble_eval_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/reuse REASSEMBLE evaluation with all viewer assets enabled."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--h5-path", help="Path to one REASSEMBLE .h5 trajectory.")
    input_group.add_argument("--data-root", help="Directory containing REASSEMBLE .h5 trajectories.")
    parser.add_argument(
        "--output-dir",
        default="REASSEMBLE__outputs_persistence",
        help="Directory for viewer-ready evaluation outputs.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of selected peaks to evaluate.")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Selectors to compare.",
    )
    parser.add_argument(
        "--segment-level",
        choices=["high", "low", "both"],
        default="low",
        help="Which failure segment level to score.",
    )
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default="persistence",
        help="Metric proposal ranking used by the viewer output. Defaults to persistence.",
    )
    parser.add_argument(
        "--evaluation-unit",
        choices=["failed_high_level_action", "successful_high_level_action"],
        default="failed_high_level_action",
        help="Which high-level demos to prepare for the viewer.",
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
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for folder evaluation.")
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Optional cap on included high-level demos across all input files.",
    )
    parser.add_argument(
        "--min-demo-duration-sec",
        type=float,
        default=None,
        help="Only include high-level demos at least this long.",
    )
    parser.add_argument(
        "--max-demo-duration-sec",
        type=float,
        default=None,
        help="Only include high-level demos at most this long.",
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
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = run_reassemble_eval_pipeline(
        h5_path=Path(args.h5_path) if args.h5_path is not None else None,
        data_root=Path(args.data_root) if args.data_root is not None else None,
        output_dir=output_dir,
        top_k=args.top_k,
        segment_level=args.segment_level,
        config_path=args.config_path,
        robot_downsample_hz=args.robot_downsample_hz,
        camera_names=["hand", "hama1", "hama2"],
        include_event_camera=False,
        extract_frames=True,
        frame_cache_dir=args.frame_cache_dir,
        max_files=args.max_files,
        selectors=args.selectors,
        metric_ranking=args.metric_ranking,
        evaluation_unit=args.evaluation_unit,
        max_demos=args.max_demos,
        min_demo_duration_sec=args.min_demo_duration_sec,
        max_demo_duration_sec=args.max_demo_duration_sec,
        plot_ecod_aggregates=True,
        resume=args.resume,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))
    print()
    print(
        "Open the viewer with: "
        f"streamlit run pathologist/reassemble/viewer_app.py -- --output-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
