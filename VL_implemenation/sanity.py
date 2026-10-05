"""Gripper-state timing sanity check helpers."""

from __future__ import annotations

from typing import Any

from VL_implemenation.abc_data import Episode, _message_timestamp_ns


def measured_gripper_closures(
    episode: Episode,
    *,
    minimum_time_sec: float = 0.0,
    open_threshold: float = 0.8,
    midpoint_threshold: float = 0.5,
    closed_threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Find clear open-to-closed state transitions for either gripper."""

    timelines = _load_gripper_timelines(episode)
    output: list[dict[str, Any]] = []
    for gripper_name, records in timelines.items():
        armed = False
        previous_value: float | None = None
        for index, record in enumerate(records):
            timestamp = float(record["timestamp_sec"])
            value = float(record["value"])
            if timestamp < minimum_time_sec:
                armed = value >= open_threshold
                previous_value = value
                continue
            if value >= open_threshold:
                armed = True
            crossed_midpoint = (
                previous_value is not None
                and previous_value > midpoint_threshold
                and value <= midpoint_threshold
            )
            previous_value = value
            if not armed or not crossed_midpoint:
                continue
            future = records[index : index + 21]
            if future and min(float(item["value"]) for item in future) <= closed_threshold:
                output.append(
                    {
                        "timestamp_sec": timestamp,
                        "gripper_name": gripper_name,
                        "midpoint_value": value,
                    }
                )
                armed = False
    output.sort(key=lambda item: float(item["timestamp_sec"]))
    deduplicated: list[dict[str, Any]] = []
    for transition in output:
        if (
            deduplicated
            and float(transition["timestamp_sec"])
            - float(deduplicated[-1]["timestamp_sec"])
            < 0.5
        ):
            continue
        deduplicated.append(transition)
    return deduplicated


def _load_gripper_timelines(episode: Episode) -> dict[str, list[dict[str, float]]]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    topic_to_name = {
        "/left-ee-state": "left",
        "/right-ee-state": "right",
    }
    with episode.episode_path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None or summary.statistics is None:
        return {}
    origin_ns = int(summary.statistics.message_start_time)
    output: dict[str, list[dict[str, float]]] = {
        name: [] for name in topic_to_name.values()
    }
    with episode.episode_path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(
            topics=list(topic_to_name)
        ):
            position = list(getattr(decoded, "position", []))
            if not position:
                continue
            timestamp_ns = _message_timestamp_ns(decoded, int(message.log_time))
            output[topic_to_name[channel.topic]].append(
                {
                    "timestamp_sec": max(0.0, (timestamp_ns - origin_ns) / 1e9),
                    "value": float(position[0]),
                }
            )
    return output
