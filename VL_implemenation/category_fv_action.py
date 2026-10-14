"""Run the paper's FV-Action fine stage for frozen ABC category prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from VL_implemenation.abc_data import discover_episodes
from VL_implemenation.category_protocol import (
    DEFAULT_EXPERIMENT_CONFIG,
    load_experiment_config,
    render_category_prompt,
    selected_categories,
)
from VL_implemenation.coarse_to_fine import run_scan
from VL_implemenation.yesno_scan import QwenYesNoScanner, cache_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("VL_implemenation/config.json"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jpeg-workers", type=int)
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = load_experiment_config(args.experiment_config)
    categories = selected_categories(experiment, args.categories)
    episodes = discover_episodes(base_config["data_root"])
    if args.limit is not None:
        episodes = episodes[: args.limit]
    external_output = Path(base_config["output_root"]).expanduser().resolve()
    settings = experiment["fv_action"]
    scan = experiment["scan"]
    scanner: QwenYesNoScanner | None = None

    for category in categories:
        coarse_cache_root = (
            external_output / "vlm_scores_by_category" / category
        )
        missing = [
            episode
            for episode in episodes
            if not cache_paths(coarse_cache_root, episode)[0].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{category}: missing {len(missing)} coarse VLM caches"
            )
        prompt = render_category_prompt(experiment, category)
        output_root = external_output / "fv_action_by_category" / category
        scanner = run_scan(
            episodes,
            coarse_cache_root=coarse_cache_root,
            frame_cache_root=Path(base_config["frame_cache_root"]),
            output_root=output_root,
            model_path=Path(base_config["model_path"]),
            model_id=str(base_config["model_id"]),
            radius_sec=float(settings["radius_sec"]),
            candidate_count=int(settings["candidate_count"]),
            fine_count=int(settings["fine_count"]),
            alpha=float(settings["alpha"]),
            window_sec=float(scan["window_sec"]),
            batch_size=int(args.batch_size or scan["batch_size"]),
            workers=int(args.jpeg_workers or scan["jpeg_workers"]),
            queue_depth=int(args.queue_depth or scan["queue_depth"]),
            compile_model=not args.no_compile,
            resume=not args.no_resume,
            prompt=prompt,
            scanner=scanner,
            output_subdirectory=None,
            provenance={
                "category": category,
                "experiment_config_path": experiment["_config_path"],
                "experiment_config_sha256": experiment["_config_sha256"],
                "taxonomy_sha256": experiment["_taxonomy_sha256"],
            },
        )


if __name__ == "__main__":
    main()
