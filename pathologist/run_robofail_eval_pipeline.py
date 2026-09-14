"""CLI for deterministic RoboFail localization evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathologist.robofail import run_robofail_eval_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic top-K localization evaluation on RoboFail trajectories."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the RoboFail dataset root.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["sim", "real"],
        help="RoboFail split to evaluate.",
    )
    parser.add_argument("--output-dir", default="robofail_outputs", help="Directory for evaluation outputs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of selected peaks to evaluate.")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Selectors to compare. Defaults to all three: uniform, kite_flow, metric.",
    )
    parser.add_argument("--config", dest="config_path", help="Optional pathologist config JSON override.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for folder evaluation.")
    parser.add_argument(
        "--no-extract-frames",
        action="store_true",
        help="For real split only, skip video frame extraction and visual-flow scoring.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write raw/smoothed ECOD aggregate plots and local-maxima JSON under ecod_aggregates/.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs and running MAE.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_robofail_eval_pipeline(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        output_dir=args.output_dir,
        top_k=args.top_k,
        selectors=args.selectors,
        config_path=args.config_path,
        max_episodes=args.max_episodes,
        extract_frames=not args.no_extract_frames,
        plot_ecod_aggregates=args.plot,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
