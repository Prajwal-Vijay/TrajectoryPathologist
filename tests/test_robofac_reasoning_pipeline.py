"""Tests for the RoboFAC explanation and task-planning evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathologist.qa.robofac import RoboFACQAItem
from pathologist.qa.robofac_client import StaticRoboFACQAClient
from pathologist.qa.robofac_reasoning import (
    OpenAIRoboFACJudgeClient,
    StaticRoboFACJudgeClient,
    build_reasoning_prompt,
    build_robofac_judge_prompt,
    parse_robofac_judge_response,
    validate_reasoning_question_types,
)
from pathologist.run_robofac_reasoning_pipeline import (
    aggregate_reasoning_stats,
    build_arg_parser,
    prediction_id,
    run_robofac_reasoning_pipeline,
    select_reasoning_work_items,
)
from pathologist.schemas import BundleFormat, CameraFrameRef, TrajectoryBundle


VALID_JUDGE_RESPONSE = json.dumps(
    {
        "criteria": {
            "correctness": {"score": 4, "explanation": "Mostly correct."},
            "relevance": {"score": 5, "explanation": "Directly relevant."},
            "completeness": {"score": 3, "explanation": "One detail is absent."},
        }
    }
)


def _item(video_id: str, question_type: str) -> RoboFACQAItem:
    return RoboFACQAItem(
        video_id=video_id,
        video_rel_path=f"Task/view0/{video_id}.mp4",
        task="Task",
        question_type=question_type,
        question=f"<image>\nQuestion for {question_type}?",
        reference_answer=f"Reference for {question_type}.",
    )


def test_reasoning_question_types_are_strictly_limited() -> None:
    assert validate_reasoning_question_types(["Task planning"]) == ["Task planning"]
    assert validate_reasoning_question_types(
        ["Failure explanation", "Task planning"]
    ) == ["Failure explanation", "Task planning"]

    with pytest.raises(ValueError, match="Failure locating"):
        validate_reasoning_question_types(["Failure locating"])
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--annotations", "annotations", "--question-types", "Failure locating"]
        )
    args = build_arg_parser().parse_args(
        ["--annotations", "annotations", "--metric-ranking", "persistence"]
    )
    assert args.metric_ranking == "persistence"


def test_reasoning_prompt_contains_only_basic_instruction_and_question() -> None:
    prompt = build_reasoning_prompt("<image>\nWhy did the task fail?")

    assert prompt == (
        "Look at the provided frames from the robot task execution, shown in "
        "chronological order, and answer the question.\n\n"
        "Question: Why did the task fail?\n\n"
        "Answer:"
    )
    assert "timestamp" not in prompt
    assert "frame_index" not in prompt
    assert "robot profile" not in prompt.lower()
    assert "metric" not in prompt.lower()


def test_robofac_appendix_judge_prompt_and_normalization() -> None:
    prompt = build_robofac_judge_prompt("Question?", "Reference.", "Prediction.")
    result = parse_robofac_judge_response(f"```json\n{VALID_JUDGE_RESPONSE}\n```")

    assert "Question: Question?" in prompt
    assert "Reference answer: Reference." in prompt
    assert "Model's response: Prediction." in prompt
    assert "score from 0 to 5" in prompt
    assert result.average_score == 4.0
    assert result.normalized_score == 80.0
    assert result.criteria["correctness"]["score"] == 4


def test_judge_client_uses_appendix_request_parameters() -> None:
    client = OpenAIRoboFACJudgeClient(
        api_key="test-key",
        model="gpt-4o",
        temperature=0.2,
        top_p=1.0,
    )

    request_body = client._client._build_request_body(prompt="Judge this.", frames=[])

    assert request_body["model"] == "gpt-4o"
    assert request_body["temperature"] == 0.2
    assert request_body["top_p"] == 1.0


def test_judge_parser_rejects_non_integer_or_missing_scores() -> None:
    payload = json.loads(VALID_JUDGE_RESPONSE)
    payload["criteria"]["correctness"]["score"] = 4.5
    with pytest.raises(ValueError, match="integer"):
        parse_robofac_judge_response(json.dumps(payload))

    del payload["criteria"]["relevance"]
    with pytest.raises(ValueError, match="relevance"):
        parse_robofac_judge_response(json.dumps(payload))


def test_seeded_sampling_caps_each_selected_question_type() -> None:
    items = [
        _item(f"video-{index}", question_type)
        for index in range(6)
        for question_type in ("Failure explanation", "Task planning")
    ]
    first, first_counts = select_reasoning_work_items(
        items,
        question_types=["Failure explanation", "Task planning"],
        video_sample_seed=42,
        max_videos=None,
        max_questions_per_type=3,
    )
    second, second_counts = select_reasoning_work_items(
        items,
        question_types=["Failure explanation", "Task planning"],
        video_sample_seed=42,
        max_videos=None,
        max_questions_per_type=3,
    )

    assert [video_id for video_id, _ in first] == [video_id for video_id, _ in second]
    assert first_counts == second_counts == {
        "Failure explanation": 3,
        "Task planning": 3,
    }
    assert all(len(video_items) == 2 for _, video_items in first)


def test_reasoning_stats_use_normalized_scores_and_criteria() -> None:
    item = _item("video-1", "Failure explanation")
    row_id = prediction_id(item, "metric")
    predictions = {
        row_id: {
            "prediction_id": row_id,
            "selector_name": "metric",
            "question_type": "Failure explanation",
            "raw_prediction": "Answer.",
            "generation_error": None,
        }
    }
    result = parse_robofac_judge_response(VALID_JUDGE_RESPONSE)
    judgments = {
        row_id: {
            "prediction_id": row_id,
            "selector_name": "metric",
            "question_type": "Failure explanation",
            **result.to_dict(),
            "judge_error": None,
        }
    }

    stats = aggregate_reasoning_stats(
        predictions,
        judgments,
        ["metric"],
        ["Failure explanation"],
    )

    assert stats["metric"]["Failure explanation"]["judged_count"] == 1
    assert stats["metric"]["Failure explanation"]["mean_score"] == 80.0
    assert stats["metric"]["Failure explanation"]["criteria"]["relevance"][
        "mean_score"
    ] == 100.0


@pytest.mark.parametrize(
    "question_types",
    [
        ["Failure explanation"],
        ["Task planning"],
        ["Failure explanation", "Task planning"],
    ],
)
def test_static_end_to_end_run_and_resume_keeps_only_supported_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    question_types: list[str],
) -> None:
    annotations_dir = tmp_path / "test_qa_sim"
    annotations_dir.mkdir()
    annotations = {
        "video-1": {
            "video": "Task/view0/video-1.mp4",
            "task": "Task",
            "annos": {
                "Failure explanation": [
                    {"from": "human", "value": "<image>\nExplain the failure."},
                    {"from": "assistant", "value": "Reference explanation."},
                ],
                "Task planning": [
                    {"from": "human", "value": "<image>\nPlan the task."},
                    {"from": "assistant", "value": "1. Do the task."},
                ],
                "Failure locating": [
                    {"from": "human", "value": "<image>\nLocate it."},
                    {"from": "assistant", "value": "At grasping."},
                ],
            },
        }
    }
    annotation_path = annotations_dir / "annos_per_video_split0.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    video_path = tmp_path / "video-1.mp4"
    h5_path = tmp_path / "view0.h5"
    traj_path = tmp_path / "view0.json"
    video_path.write_bytes(b"video")
    h5_path.write_bytes(b"h5")
    traj_path.write_text("{}", encoding="utf-8")
    image_paths = []
    for index in range(5):
        image_path = tmp_path / f"frame-{index}.jpg"
        image_path.write_bytes(b"image")
        image_paths.append(image_path)
    frames = [
        CameraFrameRef(
            camera_name="rgb",
            timestamp_sec=float(index),
            image_path=image_path,
            frame_index=index,
        )
        for index, image_path in enumerate(image_paths)
    ]
    bundle = TrajectoryBundle(
        bundle_id="video-1",
        source_format=BundleFormat.CUSTOM,
        root_path=video_path,
        camera_frames=frames,
        raw_metadata={"episode_id": 1, "traj_key": "traj_1", "frame_count": 5, "fps": 1.0},
    )

    class FakeAdapter:
        def load(self, **_kwargs: object) -> TrajectoryBundle:
            return bundle

    class FakeMetricEngine:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def compute(self, _bundle: TrajectoryBundle) -> object:
            return object()

    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline._resolve_video_path",
        lambda **_kwargs: video_path,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline._resolve_trajectory_files",
        lambda **_kwargs: (h5_path, traj_path),
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.RoboFACBundleAdapter",
        FakeAdapter,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.MetricEngine",
        FakeMetricEngine,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.select_keyframes",
        lambda **_kwargs: frames,
    )
    output_dir = tmp_path / "outputs"
    common_args = {
        "annotation_path": annotation_path,
        "dataset_root": tmp_path,
        "output_dir": output_dir,
        "question_types": question_types,
        "metric_ranking": "persistence",
        "qa_client": StaticRoboFACQAClient("Generated answer."),
        "judge_client": StaticRoboFACJudgeClient(VALID_JUDGE_RESPONSE),
    }

    summary = run_robofac_reasoning_pipeline(**common_args)

    prediction_rows = [
        json.loads(line)
        for line in (output_dir / "robofac_predictions.jsonl").read_text().splitlines()
    ]
    judgment_rows = [
        json.loads(line)
        for line in (output_dir / "robofac_judgments.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((output_dir / "robofac_manifest.json").read_text())
    stats = json.loads((output_dir / "robofac_stats.json").read_text())
    expected_count = 3 * len(question_types)
    assert summary["prediction_count"] == expected_count
    assert summary["valid_judgment_count"] == expected_count
    assert summary["question_types"] == question_types
    assert summary["metric_ranking"] == "persistence"
    assert len(prediction_rows) == len(judgment_rows) == expected_count
    assert {row["question_type"] for row in prediction_rows} == set(question_types)
    assert {row["question_type"] for row in judgment_rows} == set(question_types)
    assert manifest[0]["question_types"] == question_types
    assert manifest[0]["metric_ranking"] == "persistence"
    assert all(row["frame_budget"] == 5 for row in prediction_rows)
    assert all(len(row["selected_frame_indices"]) == 5 for row in prediction_rows)
    assert all(row["temperature"] == 0.1 for row in prediction_rows)
    assert all("timestamp" not in row["prompt"] for row in prediction_rows)
    grouped_selectors: dict[tuple[str, str], set[str]] = {}
    for row in prediction_rows:
        key = (row["question_type"], row["question"])
        grouped_selectors.setdefault(key, set()).add(row["selector_name"])
    assert all(
        selector_names == {"uniform", "kite_flow", "metric"}
        for selector_names in grouped_selectors.values()
    )
    assert "Failure locating" not in json.dumps(manifest)
    assert "Failure locating" not in json.dumps(judgment_rows)
    assert "Failure locating" not in json.dumps(stats)

    class FailingAnswerClient:
        def answer(self, _prompt: str, _frames: list[CameraFrameRef]) -> str:
            raise AssertionError("resume regenerated a completed answer")

    class FailingJudgeClient:
        def judge(self, _prompt: str) -> str:
            raise AssertionError("resume rejudged a completed answer")

    resumed = run_robofac_reasoning_pipeline(
        **{
            **common_args,
            "qa_client": FailingAnswerClient(),
            "judge_client": FailingJudgeClient(),
            "resume": True,
        }
    )
    assert resumed["prediction_count"] == expected_count
    assert resumed["valid_judgment_count"] == expected_count
    assert len((output_dir / "robofac_predictions.jsonl").read_text().splitlines()) == expected_count
    assert len((output_dir / "robofac_judgments.jsonl").read_text().splitlines()) == expected_count

    different_types = (
        ["Task planning"]
        if question_types != ["Task planning"]
        else ["Failure explanation"]
    )
    with pytest.raises(ValueError, match="different question_types"):
        run_robofac_reasoning_pipeline(
            **{
                **common_args,
                "question_types": different_types,
                "resume": True,
            }
        )

    (output_dir / "robofac_judgments.jsonl").write_text("", encoding="utf-8")

    class FailingAdapter:
        def load(self, **_kwargs: object) -> TrajectoryBundle:
            raise AssertionError("judgment-only resume reloaded video telemetry")

    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.RoboFACBundleAdapter",
        FailingAdapter,
    )
    judgment_only_resume = run_robofac_reasoning_pipeline(
        **{
            **common_args,
            "qa_client": FailingAnswerClient(),
            "judge_client": StaticRoboFACJudgeClient(VALID_JUDGE_RESPONSE),
            "resume": True,
        }
    )
    assert judgment_only_resume["valid_judgment_count"] == expected_count
    assert len((output_dir / "robofac_judgments.jsonl").read_text().splitlines()) == expected_count


def test_missing_candidate_does_not_consume_max_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations_dir = tmp_path / "test_qa_sim"
    annotations_dir.mkdir()
    annotations = {
        f"video-{index}": {
            "video": f"Task/view0/video-{index}.mp4",
            "task": "Task",
            "annos": {
                "Failure explanation": [
                    {"from": "human", "value": "<image>\nExplain the failure."},
                    {"from": "assistant", "value": "Reference explanation."},
                ]
            },
        }
        for index in range(1, 4)
    }
    annotation_path = annotations_dir / "annos_per_video_split0.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    video_path = tmp_path / "video.mp4"
    h5_path = tmp_path / "trajectory.h5"
    traj_path = tmp_path / "trajectory.json"
    video_path.write_bytes(b"video")
    h5_path.write_bytes(b"h5")
    traj_path.write_text("{}", encoding="utf-8")
    frames = []
    for index in range(5):
        image_path = tmp_path / f"replacement-frame-{index}.jpg"
        image_path.write_bytes(b"image")
        frames.append(
            CameraFrameRef(
                camera_name="rgb",
                timestamp_sec=float(index),
                image_path=image_path,
                frame_index=index,
            )
        )
    bundle = TrajectoryBundle(
        bundle_id="replacement-test",
        source_format=BundleFormat.CUSTOM,
        root_path=video_path,
        camera_frames=frames,
        raw_metadata={"frame_count": 5, "fps": 1.0},
    )

    class FakeAdapter:
        def load(self, **_kwargs: object) -> TrajectoryBundle:
            return bundle

    def resolve_trajectory_files(**kwargs: object) -> tuple[Path, Path]:
        if "video-1" in str(kwargs["video_rel_path"]):
            raise FileNotFoundError("missing candidate telemetry")
        return h5_path, traj_path

    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline._resolve_video_path",
        lambda **_kwargs: video_path,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline._resolve_trajectory_files",
        resolve_trajectory_files,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.RoboFACBundleAdapter",
        FakeAdapter,
    )
    monkeypatch.setattr(
        "pathologist.run_robofac_reasoning_pipeline.select_keyframes",
        lambda **_kwargs: frames,
    )
    output_dir = tmp_path / "replacement-outputs"

    summary = run_robofac_reasoning_pipeline(
        annotation_path=annotation_path,
        dataset_root=tmp_path,
        output_dir=output_dir,
        question_types=["Failure explanation"],
        selectors=["uniform"],
        max_videos=2,
        qa_client=StaticRoboFACQAClient("Generated answer."),
        judge_client=StaticRoboFACJudgeClient(VALID_JUDGE_RESPONSE),
    )

    manifest = json.loads((output_dir / "robofac_manifest.json").read_text())
    alignment = json.loads(
        (output_dir / "robofac_alignment_debug.json").read_text()
    )
    assert summary["selected_video_count"] == 2
    assert summary["processed_video_count"] == 2
    assert {row["video_id"] for row in manifest} == {"video-2", "video-3"}
    assert [row["video_id"] for row in alignment] == ["video-1"]
