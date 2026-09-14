"""CLI for ABC-130K annotated failure localization."""

from __future__ import annotations

import argparse
import json

from pathologist.abc130k import run_abc130k_eval_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PRISM, Uniform, and KITE at K on ABC-130K mistake intervals."
    )
    parser.add_argument(
        "--data-root",
        default="ABC-130k_subset/data/train",
        help="Root containing <task>/<episode>/{episode,annotation}.mcap.",
    )
    parser.add_argument("--output-dir", default="ABC130K_failure_eval")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "kite", "flow", "kite_flow", "prism", "metric"],
    )
    parser.add_argument(
        "--pad",
        nargs="+",
        default=[],
        choices=["kite", "flow", "kite_flow", "prism", "metric"],
        metavar="SELECTOR",
        help=(
            "Uniformly pad the named selectors to exactly --top-k frames; "
            "for example: --pad prism kite."
        ),
    )
    parser.add_argument(
        "--camera",
        "--camera-name",
        dest="camera_names",
        action="append",
        choices=["top", "left_wrist", "right_wrist"],
        help=(
            "Camera to decode/use. Can be passed multiple times. "
            "Defaults to top, left_wrist, and right_wrist."
        ),
    )
    parser.add_argument(
        "--camera-sample-hz",
        type=float,
        default=None,
        help="Optional camera downsampling rate. Default uses every native frame.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default=None,
        help=(
            "Reusable decoded-frame cache. Defaults to "
            "<output-dir>/frame_cache."
        ),
    )
    parser.add_argument("--robot-downsample-hz", type=float, default=50.0)
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default="persistence",
    )
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument(
        "--visual-flow-frame-stride",
        type=int,
        default=2,
        help="Frame stride used by PRISM's visual-flow source metric.",
    )
    parser.add_argument(
        "--kite-frame-stride",
        type=int,
        default=2,
        help="Independent frame stride used by the KITE baseline.",
    )
    parser.add_argument(
        "--robot-mask-visual-flow",
        action="store_true",
        help="Exclude zero-shot Grounding DINO + SAM2 robot masks from PRISM flow.",
    )
    parser.add_argument(
        "--robot-mask-prompt",
        default="robot arm. robot gripper.",
        help="Open-vocabulary Grounding DINO prompt used to initialize SAM2 tracking.",
    )
    parser.add_argument(
        "--robot-mask-box-threshold",
        type=float,
        default=0.35,
        help="Grounding DINO confidence threshold for robot-part boxes.",
    )
    parser.add_argument(
        "--robot-mask-dilation-px",
        type=int,
        default=8,
        help="Mask dilation radius at the configured visual-flow resolution.",
    )
    parser.add_argument(
        "--robot-mask-cache-dir",
        default=None,
        help="Optional reusable directory for binary robot masks.",
    )
    parser.add_argument(
        "--all-failure-trajectories",
        action="store_true",
        help="Evaluate every downloaded failure trajectory instead of one per task.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_abc130k_eval_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
        selectors=args.selectors,
        pad_selectors=args.pad,
        camera_names=args.camera_names,
        camera_sample_hz=args.camera_sample_hz,
        robot_downsample_hz=args.robot_downsample_hz,
        frame_cache_dir=args.frame_cache_dir,
        metric_ranking=args.metric_ranking,
        smoothing_window=args.smoothing_window,
        visual_flow_frame_stride=args.visual_flow_frame_stride,
        kite_frame_stride=args.kite_frame_stride,
        robot_mask_visual_flow=args.robot_mask_visual_flow,
        robot_mask_prompt=args.robot_mask_prompt,
        robot_mask_box_threshold=args.robot_mask_box_threshold,
        robot_mask_dilation_px=args.robot_mask_dilation_px,
        robot_mask_cache_dir=args.robot_mask_cache_dir,
        one_per_task=not args.all_failure_trajectories,
        max_episodes=args.max_episodes,
        config_path=args.config_path,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        resume=args.resume,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
