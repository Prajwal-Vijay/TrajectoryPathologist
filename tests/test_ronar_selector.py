from pathlib import Path

import pytest

from pathologist.qa import ronar_selector
from pathologist.qa.ronar_selector import (
    RonarFeatureSeries,
    positive_zscore_sum,
    select_ronar_keyevents,
    threshold_crossings,
)
from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TrajectoryBundle,
)


def test_positive_zscores_are_summed_and_nonpositive_values_are_ignored() -> None:
    features = RonarFeatureSeries(
        timestamps_sec=[0.0, 0.2, 0.4],
        values_by_feature={
            "flow:top": [0.0, 1.0, 2.0],
            "arm_motion": [0.0, 2.0, 0.0],
        },
    )

    scores = positive_zscore_sum(
        features,
        {
            "flow:top": {"mean": 0.0, "std": 1.0},
            "arm_motion": {"mean": 1.0, "std": 1.0},
        },
    )

    assert scores == pytest.approx([0.0, 2.0, 2.0])


def test_threshold_crossing_is_strict_and_resets_accumulator() -> None:
    indices, accumulators = threshold_crossings([2.0, 2.0, 2.0, 2.0], 3.0)

    assert indices == [1, 3]
    assert accumulators == pytest.approx([4.0, 4.0])
    assert threshold_crossings([3.0], 3.0)[0] == []


def test_zero_variance_features_contribute_zero() -> None:
    features = RonarFeatureSeries(
        timestamps_sec=[0.0, 0.2],
        values_by_feature={"arm_motion": [1.0, 5.0]},
    )

    assert positive_zscore_sum(
        features,
        {"arm_motion": {"mean": 1.0, "std": 0.0}},
    ) == [0.0, 0.0]


def test_joint_motion_is_aligned_on_paper_sampling_grid(tmp_path: Path) -> None:
    stream = ModalityStream(
        name="proprioception",
        modality=Modality.PROPRIOCEPTION,
        timestamps_sec=[0.0, 0.2, 0.4],
        records=[
            {"joint_positions": [0.0, 0.0]},
            {"joint_positions": [1.0, 2.0]},
            {"joint_positions": [2.0, 4.0]},
        ],
    )
    bundle = TrajectoryBundle(
        bundle_id="aligned",
        source_format=BundleFormat.CUSTOM,
        root_path=tmp_path,
        duration_sec=0.4,
        streams=[stream],
    )

    features = ronar_selector.extract_ronar_features(
        bundle,
        camera_names=[],
        sample_period_sec=0.2,
    )

    assert features.timestamps_sec == pytest.approx([0.0, 0.2, 0.4])
    assert features.values_by_feature["arm_motion"] == pytest.approx(
        [0.0, 3.0, 3.0]
    )


def test_ronar_k_downselects_discrete_overcomplete_crossings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _frame_bundle(tmp_path, 12)
    features = RonarFeatureSeries(
        timestamps_sec=[frame.timestamp_sec for frame in bundle.camera_frames],
        values_by_feature={"arm_motion": [float(index) for index in range(12)]},
    )
    monkeypatch.setattr(
        ronar_selector,
        "extract_ronar_features",
        lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        ronar_selector,
        "positive_zscore_sum",
        lambda *args, **kwargs: [1.0] * 12,
    )

    selection = select_ronar_keyevents(bundle, camera_names=["top"], frame_budget=5)

    assert len(selection.events) == 5
    assert selection.metadata["protocol"] == "per_trajectory_exact_k"
    assert selection.metadata["genuine_crossing_count"] == 6
    assert selection.metadata["downselected_crossing_count"] == 1
    assert selection.metadata["uniform_padded_event_count"] == 0
    assert all(
        event.selection_source == "ronar_k_crossing"
        for event in selection.events
    )


def test_ronar_k_records_uniform_padding_when_crossings_are_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _frame_bundle(tmp_path, 8)
    features = RonarFeatureSeries(
        timestamps_sec=[frame.timestamp_sec for frame in bundle.camera_frames],
        values_by_feature={"arm_motion": [0.0] * 8},
    )
    monkeypatch.setattr(
        ronar_selector,
        "extract_ronar_features",
        lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        ronar_selector,
        "positive_zscore_sum",
        lambda *args, **kwargs: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )

    selection = select_ronar_keyevents(bundle, frame_budget=5)

    assert len(selection.events) == 5
    assert selection.metadata["genuine_crossing_count"] == 2
    assert selection.metadata["uniform_padded_event_count"] == 3
    assert sum(
        event.selection_source == "uniform_padding"
        for event in selection.events
    ) == 3


def test_strict_ronar_is_variable_count_and_never_pads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _frame_bundle(tmp_path, 6)
    features = RonarFeatureSeries(
        timestamps_sec=[frame.timestamp_sec for frame in bundle.camera_frames],
        values_by_feature={"arm_motion": [0.0] * 6},
    )
    monkeypatch.setattr(
        ronar_selector,
        "extract_ronar_features",
        lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        ronar_selector,
        "positive_zscore_sum",
        lambda *args, **kwargs: [20.0] * 6,
    )

    selection = select_ronar_keyevents(bundle, threshold=80.0)

    assert len(selection.events) == 1
    assert selection.metadata["protocol"] == "published_threshold"
    assert selection.metadata["uniform_padded_event_count"] == 0
    assert selection.metadata["downselected_crossing_count"] == 0


def _frame_bundle(tmp_path: Path, count: int) -> TrajectoryBundle:
    frames = [
        CameraFrameRef(
            camera_name="top",
            timestamp_sec=index * 0.2,
            image_path=tmp_path / f"{index}.png",
            frame_index=index,
        )
        for index in range(count)
    ]
    return TrajectoryBundle(
        bundle_id="frames",
        source_format=BundleFormat.CUSTOM,
        root_path=tmp_path,
        duration_sec=frames[-1].timestamp_sec,
        camera_frames=frames,
    )
