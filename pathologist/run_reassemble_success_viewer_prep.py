"""Prepare successful REASSEMBLE high-level demos for the local viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathologist.reassemble import run_reassemble_eval_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare successful REASSEMBLE high-level action demos for the Streamlit viewer."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--h5-path", help="Path to one REASSEMBLE .h5 trajectory.")
    input_group.add_argument("--data-root", help="Directory containing REASSEMBLE .h5 trajectories.")
    parser.add_argument(
        "--output-dir",
        default="REASSEMBLE_Successful",
        help="Directory for viewer-ready successful-demo outputs.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of selected peaks per selector.")
    parser.add_argument(
        "--max-demos",
        type=int,
        default=100,
        help="Maximum number of successful high-level demos to include.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Selectors to compare.",
    )
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default="persistence",
        help="Metric proposal ranking used by the viewer output.",
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
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on input H5 files.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing successful-demo outputs in --output-dir.",
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
        segment_level="low",
        config_path=args.config_path,
        robot_downsample_hz=args.robot_downsample_hz,
        camera_names=["hand", "hama1", "hama2"],
        include_event_camera=False,
        extract_frames=True,
        frame_cache_dir=args.frame_cache_dir,
        max_files=args.max_files,
        selectors=args.selectors,
        metric_ranking=args.metric_ranking,
        evaluation_unit="successful_high_level_action",
        max_demos=args.max_demos,
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
