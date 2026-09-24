"""CLI for ViFailBack exact-frame failure localization."""

from __future__ import annotations

import argparse
import json

from pathologist.vifailback import run_vifailback_localization_eval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PRISM, Random, and KITE at K=2 against ViFailBack's "
            "exact 25-Hz avoid keyframes."
        )
    )
    parser.add_argument(
        "--data-root",
        default="/media/helen/28TB_VOL4/ViFailback-Dataset",
    )
    parser.add_argument("--output-dir", default="ViFailBack_localization_k2")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Savitzky-Golay window for PRISM's 25-Hz aggregate surprise curve.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["random", "kite_flow", "metric"],
        choices=[
            "random",
            "uniform",
            "kite",
            "flow",
            "kite_flow",
            "prism",
            "metric",
        ],
    )
    parser.add_argument(
        "--tolerance-frames",
        nargs="+",
        type=int,
        default=[0, 6, 12, 25],
        help=(
            "Exact source-frame tolerances used for Hit@K. Defaults correspond "
            "to exact, 0.24 s, 0.48 s, and 1.0 s at 25 Hz."
        ),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--frame-cache-dir", default=None)
    parser.add_argument("--keep-frame-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_vifailback_localization_eval(
        data_root=args.data_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
        smoothing_window=args.smoothing_window,
        selectors=args.selectors,
        tolerance_frames=args.tolerance_frames,
        max_episodes=args.max_episodes,
        config_path=args.config_path,
        frame_cache_dir=args.frame_cache_dir,
        keep_frame_cache=args.keep_frame_cache,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        resume=args.resume,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
