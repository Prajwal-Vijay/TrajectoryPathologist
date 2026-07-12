"""Tests for RoboFAC pipeline download failure handling."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from pathologist.run_robofac_qa_pipeline import (
    RemoteFileUnavailableError,
    _download_if_missing,
    _resolve_annotation_paths,
    _resolve_trajectory_files,
    _resolve_video_path,
    _selectors_available_for_bundle,
    _video_items_for_sampling,
)


def test_download_if_missing_removes_partial_file_and_raises_remote_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "missing.mp4"

    def fail_download(_url: str, path: Path) -> None:
        Path(path).write_text("partial", encoding="utf-8")
        raise urllib.error.HTTPError(
            url="https://example.com/missing.mp4",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlretrieve", fail_download)

    with pytest.raises(RemoteFileUnavailableError, match="HTTP 404"):
        _download_if_missing(
            url="https://example.com/missing.mp4",
            destination=destination,
            label="video",
        )

    assert not destination.exists()


def test_resolve_annotation_paths_accepts_folder_and_sorts_naturally(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "test_qa_sim"
    annotations_dir.mkdir()
    split10 = annotations_dir / "annos_per_video_split10.json"
    split2 = annotations_dir / "annos_per_video_split2.json"
    split10.write_text("{}", encoding="utf-8")
    split2.write_text("{}", encoding="utf-8")
    (annotations_dir / "training_qa.json").write_text("[]", encoding="utf-8")

    assert _resolve_annotation_paths(annotations_dir) == [
        split2.resolve(),
        split10.resolve(),
    ]
    assert _resolve_annotation_paths(split10) == [split10.resolve()]


def test_resolve_video_path_uses_full_dataset_root_for_sim_and_realworld(tmp_path: Path) -> None:
    dataset_root = tmp_path / "RoboFAC-dataset"
    sim_video = dataset_root / "simulation_data" / "Task" / "view0" / "video-uuid.mp4"
    real_video = (
        dataset_root
        / "realworld_data"
        / "so100_task_error"
        / "videos"
        / "chunk-000"
        / "observation.images.above"
        / "episode_000000.mp4"
    )
    sim_video.parent.mkdir(parents=True)
    real_video.parent.mkdir(parents=True)
    sim_video.write_text("sim", encoding="utf-8")
    real_video.write_text("real", encoding="utf-8")

    assert _resolve_video_path(
        video_id="video-uuid",
        video_rel_path="Task/view0/video-uuid.mp4",
        video_root=dataset_root,
        explicit_video_path=None,
    ) == sim_video.resolve()
    assert _resolve_video_path(
        video_id="annotation-uuid",
        video_rel_path=(
            "so100_task_error/videos/chunk-000/"
            "observation.images.above/episode_000000.mp4"
        ),
        video_root=dataset_root,
        explicit_video_path=None,
    ) == real_video.resolve()


def test_resolve_trajectory_files_finds_leaf_h5_json_pair_under_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "RoboFAC-dataset"
    leaf = dataset_root / "simulation_data" / "LiftPegUpright-v1" / "view0"
    leaf.mkdir(parents=True)
    h5_path = leaf / "view0.h5"
    json_path = leaf / "view0.json"
    h5_path.write_text("h5", encoding="utf-8")
    json_path.write_text("json", encoding="utf-8")

    assert _resolve_trajectory_files(
        video_rel_path="LiftPegUpright-v1/view0/video-uuid.mp4",
        explicit_h5_path=None,
        explicit_traj_json_path=None,
        data_root=dataset_root,
        download_missing=False,
        hf_base_url=None,
        download_dir=tmp_path / "downloads",
    ) == (h5_path.resolve(), json_path.resolve())


def test_resolve_trajectory_files_can_return_none_for_video_only_data(tmp_path: Path) -> None:
    assert _resolve_trajectory_files(
        video_rel_path="so100_task_error/videos/chunk-000/observation.images.above/episode_000000.mp4",
        explicit_h5_path=None,
        explicit_traj_json_path=None,
        data_root=tmp_path,
        download_missing=False,
        hf_base_url=None,
        download_dir=tmp_path / "downloads",
        allow_missing=True,
    ) is None


def test_metric_selector_is_omitted_for_video_only_bundle() -> None:
    assert _selectors_available_for_bundle(
        ["uniform", "kite_flow", "metric"],
        {"video_only": True},
    ) == ["uniform", "kite_flow"]
    assert _selectors_available_for_bundle(
        ["uniform", "metric"],
        {},
    ) == ["uniform", "metric"]


def test_video_items_for_sampling_preserves_order_by_default() -> None:
    items_by_video = {
        "video-1": ["a"],
        "video-2": ["b"],
        "video-3": ["c"],
    }

    assert _video_items_for_sampling(
        items_by_video,
        shuffle_videos=False,
        video_sample_seed=None,
    ) == [
        ("video-1", ["a"]),
        ("video-2", ["b"]),
        ("video-3", ["c"]),
    ]


def test_video_items_for_sampling_uses_reproducible_seed() -> None:
    items_by_video = {
        f"video-{index}": [index]
        for index in range(10)
    }

    first = _video_items_for_sampling(
        items_by_video,
        shuffle_videos=True,
        video_sample_seed=123,
    )
    second = _video_items_for_sampling(
        items_by_video,
        shuffle_videos=False,
        video_sample_seed=123,
    )

    assert first == second
    assert first != list(items_by_video.items())
