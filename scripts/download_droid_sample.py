#!/usr/bin/env python3
"""Download a sampled subset of raw DROID episodes with videos.

The script expects droid_200_episode_dirs.json produced from the public GCS
listing and copies each sampled episode with rclone. It is resumable: completed
episode URIs are recorded in droid_200_downloaded.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


EXCLUDE_PATTERNS = (
    "**/*.svo",
    "**/*.SVO",
    "**/*stereo*.mp4",
    "**/*STEREO*.mp4",
)


def _gcs_to_rclone(uri: str) -> str:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    return ":gcs:" + uri[len("gs://") :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="droid_200_episode_dirs.json")
    parser.add_argument("--output-dir", default="DROID_200_raw")
    parser.add_argument("--done-file", default="droid_200_downloaded.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    done_path = Path(args.done_file)

    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    if args.limit is not None:
        episodes = episodes[: args.limit]

    done = set()
    if done_path.exists():
        done = {line.strip() for line in done_path.read_text().splitlines() if line.strip()}

    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for index, episode in enumerate(episodes, start=1):
        source_uri = episode["source_uri"]
        if source_uri in done:
            skipped += 1
            print(f"[{index}/{len(episodes)}] already done: {source_uri}", flush=True)
            continue

        src = _gcs_to_rclone(source_uri)
        dst = Path(episode["local_dir"])
        dst.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rclone",
            "copy",
            "--gcs-anonymous",
            "--transfers",
            "8",
            "--checkers",
            "16",
            "--progress",
        ]
        for pattern in EXCLUDE_PATTERNS:
            cmd.extend(["--exclude", pattern])
        cmd.extend([src, str(dst)])

        print(f"[{index}/{len(episodes)}] copying {source_uri}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue

        subprocess.run(cmd, check=True)
        with done_path.open("a") as done_file:
            done_file.write(source_uri + "\n")
        copied += 1

    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "output_dir": str(output_root),
                "copied": copied,
                "skipped": skipped,
                "requested": len(episodes),
                "done_file": str(done_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
