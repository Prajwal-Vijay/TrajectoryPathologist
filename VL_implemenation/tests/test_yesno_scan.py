from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from VL_implemenation.abc_data import Episode
from VL_implemenation.yesno_scan import (
    FRAME_RATE,
    LoadedEpisode,
    QwenYesNoScanner,
    Window,
    build_windows,
    load_episode_frames,
)


MODEL_PATH = Path("VL_implemenation/model")
RUN_GPU = os.environ.get("RUN_QWEN_GPU_TESTS") == "1"


def test_windows_cover_last_frame_and_use_even_frame_count() -> None:
    times = [index / FRAME_RATE for index in range(23)]

    windows = build_windows(times, window_frames=10, stride_frames=5)

    assert [window.indices[0] for window in windows] == [0, 5, 10, 13]
    assert windows[-1].indices[-1] == len(times) - 1
    assert all(len(window.indices) == 10 for window in windows)
    assert all(len(window.indices) % 2 == 0 for window in windows)
    for window in windows:
        assert window.center_sec == pytest.approx(
            np.mean([times[index] for index in window.indices])
        )


def test_short_clip_is_padded_to_even_fixed_window() -> None:
    times = [0.0, 0.2, 0.4]

    windows = build_windows(times, window_frames=10, stride_frames=5)

    assert len(windows) == 1
    assert windows[0].indices == (0, 1, 2, 2, 2, 2, 2, 2, 2, 2)
    assert windows[0].end_sec == pytest.approx(0.4)


def test_mosaic_shape_and_nearest_wrist_frame_matching(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    task_name = "task"
    bundle_id = "bundle"
    colors = {
        "top": (10, 20, 30),
        "left_wrist": (40, 50, 60),
        "right_wrist": (70, 80, 90),
    }
    timestamps = {
        "top": [0.0, 0.2],
        "left_wrist": [0.01, 0.21],
        "right_wrist": [0.02, 0.19],
    }
    for camera, camera_times in timestamps.items():
        directory = tmp_path / task_name / bundle_id / f"{camera}_5hz"
        directory.mkdir(parents=True)
        for index, timestamp in enumerate(camera_times):
            image = np.full((32, 48, 3), colors[camera], dtype=np.uint8)
            assert cv2.imwrite(
                str(directory / f"{camera}_{index:03d}_t{timestamp:.3f}.jpg"),
                image,
            )
    episode = Episode(
        task_name=task_name,
        bundle_id=bundle_id,
        episode_path=tmp_path / "episode.mcap",
        annotation_path=tmp_path / "annotation.mcap",
        duration_sec=0.2,
        ground_intervals=(),
    )

    loaded = load_episode_frames(
        episode,
        frame_cache_root=tmp_path,
        frame_rate=FRAME_RATE,
        window_frames=10,
        stride_frames=5,
        workers=2,
    )

    assert loaded.frames.shape == (2, 256, 960, 3)
    # Input colors are BGR; the scanner converts the complete mosaic to RGB.
    assert loaded.frames[1, 0, 320].tolist() == [60, 50, 40]
    assert loaded.frames[1, 0, 640].tolist() == [90, 80, 70]
    assert loaded.times.tolist() == pytest.approx([0.0, 0.2])


def test_yes_and_no_are_single_tokens() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("Local Qwen model is unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    assert len(tokenizer.encode("Yes", add_special_tokens=False)) == 1
    assert len(tokenizer.encode("No", add_special_tokens=False)) == 1


@pytest.fixture(scope="module")
def gpu_scanner() -> QwenYesNoScanner:
    if not RUN_GPU:
        pytest.skip("Set RUN_QWEN_GPU_TESTS=1 to run Qwen GPU tests")
    return QwenYesNoScanner(
        model_path=MODEL_PATH,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        frame_rate=FRAME_RATE,
        window_frames=10,
        batch_size=4,
    )


def _gpu_loaded_episode() -> LoadedEpisode:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(40, 256, 960, 3), dtype=np.uint8)
    times = np.arange(40, dtype=np.float64) / FRAME_RATE
    windows = [
        Window(
            indices=tuple(range(first, first + 10)),
            start_sec=float(times[first]),
            end_sec=float(times[first + 9]),
            center_sec=float(np.mean(times[first : first + 10])),
        )
        for first in (0, 10, 20, 30)
    ]
    episode = Episode(
        task_name="gpu_test",
        bundle_id="gpu_test",
        episode_path=Path("episode.mcap"),
        annotation_path=Path("annotation.mcap"),
        duration_sec=8.0,
        ground_intervals=(),
    )
    return LoadedEpisode(episode, frames, times, windows, 0.0)


@pytest.mark.gpu
def test_batched_margins_match_individual_windows(
    gpu_scanner: QwenYesNoScanner,
) -> None:
    loaded = _gpu_loaded_episode()

    batched = gpu_scanner.score(loaded).margin
    individual = np.asarray(
        [
            gpu_scanner.score(
                LoadedEpisode(
                    loaded.episode,
                    loaded.frames,
                    loaded.times,
                    [window],
                    0.0,
                )
            ).margin[0]
            for window in loaded.windows
        ]
    )

    assert np.max(np.abs(batched - individual)) < 0.05


@pytest.mark.gpu
def test_last_token_logits_match_full_forward(
    gpu_scanner: QwenYesNoScanner,
) -> None:
    torch = gpu_scanner.torch
    loaded = _gpu_loaded_episode()
    video = loaded.frames[list(loaded.windows[0].indices)]
    metadata = {
        "fps": FRAME_RATE,
        "frames_indices": list(range(10)),
        "total_num_frames": 10,
        "duration": 2.0,
    }
    inputs = gpu_scanner.processor(
        text=[gpu_scanner.prompt_text],
        videos=[video],
        video_metadata=[metadata],
        do_resize=False,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    ).to(gpu_scanner.device)

    with torch.inference_mode():
        hidden = gpu_scanner.model.model(
            **inputs, use_cache=False
        ).last_hidden_state[:, -1].float()
        kept = hidden @ gpu_scanner.choice_weight.T
        full = torch.nn.functional.linear(
            hidden,
            gpu_scanner.model.lm_head.weight.float(),
        )
    kept_margin = kept[:, 0] - kept[:, 1]
    full_margin = full[:, gpu_scanner.yes_id] - full[:, gpu_scanner.no_id]

    assert torch.max(torch.abs(kept_margin - full_margin)).item() < 0.01
