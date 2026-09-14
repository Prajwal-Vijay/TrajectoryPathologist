"""REASSEMBLE event-type classification: annotation parsing, prompting, and scoring.

The hand annotations in ``annotations.txt`` are free text, so parsing normalizes the
spelling variants the annotator used ("pull"/"active pull", "release"/"gripper open",
"appraoch") into the seven canonical classes the VLM chooses between. Every rejected or
repaired line is reported so the parse stays auditable rather than silently dropping
ground truth.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


EVENT_LABELS: tuple[str, ...] = (
    "approach",
    "first contact",
    "gripper close",
    "gripper open",
    "active push",
    "active pull",
    "engaged rotation",
)

EVENT_LABEL_DESCRIPTIONS: dict[str, str] = {
    "approach": (
        "the arm is moving from where it was toward the region where the task must be "
        "completed, with background motion visible; it has not touched the object yet"
    ),
    "first contact": (
        "the first moment the arm touches the object, or the moment a held object "
        "touches the ground"
    ),
    "gripper close": "the moment the gripper fingers close on the object",
    "gripper open": (
        "the moment the gripper fingers open and let go (also called release)"
    ),
    "active push": (
        "the gripper is touching the object and the object is visibly moving inward, "
        "for example being pushed or inserted into place"
    ),
    "active pull": (
        "the object is held and visibly moving outward, for example being pulled out, "
        "but has not fully come free yet"
    ),
    "engaged rotation": (
        "the gripper is engaged with the object and rotating it, often while pressing "
        "inward; these spans are typically long"
    ),
}

# Free-text spellings the annotator used, mapped onto the canonical label set. Matched
# longest-first so "active push" wins over "push" and "gripper close" over "close".
LABEL_ALIASES: dict[str, str] = {
    "approach": "approach",
    "appraoch": "approach",
    "first contact": "first contact",
    "firstcontact": "first contact",
    "first-contact": "first contact",
    "gripper close": "gripper close",
    "gripper closed": "gripper close",
    "gripper closes": "gripper close",
    "close gripper": "gripper close",
    "closed gripper": "gripper close",
    "gripper open": "gripper open",
    "gripper opens": "gripper open",
    "open gripper": "gripper open",
    "release": "gripper open",
    "active push": "active push",
    "push": "active push",
    "insertion": "active push",
    "insert": "active push",
    "active pull": "active pull",
    "pull": "active pull",
    "engaged rotation": "engaged rotation",
    "engaged rotations": "engaged rotation",
}

# Present in the annotations but held out of the label set: only two instances exist,
# which is too few to measure in either direction.
EXCLUDED_ANNOTATION_LABELS: dict[str, str] = {
    "stalling": "stall",
    "stall": "stall",
}

# Annotator commentary rather than events.
NON_EVENT_ANNOTATION_PHRASES: tuple[str, ...] = ("weird trajectory", "weird demo")

DEMO_HEADER_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_high_\d+)\b")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
DEFAULT_TOLERANCES_SEC: tuple[float, ...] = (1.0, 2.0, 3.0)
# Annotations this close together are treated as one simultaneous moment.
TIE_EPSILON_SEC = 1e-6

# Cameras are held fixed across every selector and peak. The recorded per-peak camera is
# deliberately ignored: it was never held constant, and uniform drew the wrist view four
# times as often as kite_flow, which would show up as a selector difference.
DEFAULT_CAMERAS: tuple[str, ...] = ("hama1", "hand")
CAMERA_DESCRIPTIONS: dict[str, str] = {
    "hama1": "external camera viewing the whole workspace",
    "hama2": "second external camera viewing the whole workspace",
    "hand": "camera mounted on the robot wrist, giving a close-up of the gripper",
}

# Selector directory names written by the REASSEMBLE peak-selection run.
SELECTOR_NAMES: tuple[str, ...] = ("uniform", "kite_flow", "metric")
SELECTOR_ALIASES: dict[str, str] = {
    "flow": "kite_flow",
    "kite": "kite_flow",
    "ecod": "metric",
    "prism": "metric",
}


def resolve_selector_names(values: Any) -> list[str]:
    """Normalize requested selectors into a deduped list in canonical order.

    Accepts ``None`` or ``"all"`` for every selector, space- or comma-separated tokens,
    and the aliases in ``SELECTOR_ALIASES``. Canonical ordering keeps the per-selector
    columns in the stats comparable across runs.
    """

    if values is None:
        return list(SELECTOR_NAMES)
    if isinstance(values, str):
        values = [values]
    requested: set[str] = set()
    for value in values:
        for token in str(value).replace(",", " ").split():
            name = token.strip().lower()
            if not name:
                continue
            if name == "all":
                requested.update(SELECTOR_NAMES)
                continue
            name = SELECTOR_ALIASES.get(name, name)
            if name not in SELECTOR_NAMES:
                allowed = ", ".join(["all", *SELECTOR_NAMES, *sorted(SELECTOR_ALIASES)])
                raise ValueError(f"Unknown selector {token!r}. Choose from: {allowed}.")
            requested.add(name)
    if not requested:
        raise ValueError("At least one selector is required.")
    return [name for name in SELECTOR_NAMES if name in requested]


@dataclass(frozen=True)
class AnnotatedEvent:
    """One hand-annotated event on a demo-relative clock."""

    demo_id: str
    label: str
    start_sec: float
    end_sec: float
    reference_sec: float
    raw_line: str

    @property
    def is_instant(self) -> bool:
        return self.end_sec <= self.start_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "label": self.label,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "reference_sec": self.reference_sec,
            "is_instant": self.is_instant,
            "raw_line": self.raw_line,
        }


def parse_event_annotations(
    path: str | Path,
) -> tuple[dict[str, list[AnnotatedEvent]], list[dict[str, Any]]]:
    """Parse the free-text annotation file into per-demo canonical events.

    Returns the events keyed by demo id (including demos whose header exists but carry no
    events) alongside a list of notes describing every line that was skipped or repaired.
    """

    text = Path(path).expanduser().read_text(encoding="utf-8")
    events_by_demo: dict[str, list[AnnotatedEvent]] = {}
    notes: list[dict[str, Any]] = []
    current_demo: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        header = DEMO_HEADER_PATTERN.match(line)
        if header is not None:
            current_demo = header.group(1)
            events_by_demo.setdefault(current_demo, [])
            continue
        if current_demo is None:
            continue
        event, note = _parse_event_line(current_demo, line)
        if event is not None:
            events_by_demo[current_demo].append(event)
        if note is not None:
            notes.append({"line_number": line_number, "demo_id": current_demo, **note})
    for events in events_by_demo.values():
        events.sort(key=lambda event: (event.reference_sec, event.label))
    return events_by_demo, notes


def _parse_event_line(
    demo_id: str, line: str
) -> tuple[AnnotatedEvent | None, dict[str, Any] | None]:
    lowered = line.lower()
    for phrase in NON_EVENT_ANNOTATION_PHRASES:
        if phrase in lowered:
            return None, {"raw_line": line, "reason": "annotator_comment"}
    label, remainder = _match_label(lowered)
    if label is None:
        excluded = _match_excluded_label(lowered)
        reason = "excluded_label" if excluded else "unrecognized_label"
        note = {"raw_line": line, "reason": reason}
        if excluded:
            note["excluded_label"] = excluded
        return None, note
    numbers = [float(value) for value in NUMBER_PATTERN.findall(remainder)]
    if not numbers:
        return None, {"raw_line": line, "reason": "no_timestamps", "label": label}
    note: dict[str, Any] | None = None
    if len(numbers) >= 3:
        start_sec, end_sec, reference_sec = numbers[0], numbers[1], numbers[2]
    elif len(numbers) == 2:
        start_sec, end_sec = numbers[0], numbers[1]
        reference_sec = (start_sec + end_sec) / 2.0
        note = {
            "raw_line": line,
            "reason": "range_without_peak_used_midpoint",
            "label": label,
            "reference_sec": reference_sec,
        }
    else:
        start_sec = end_sec = reference_sec = numbers[0]
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
        note = {"raw_line": line, "reason": "reversed_range_swapped", "label": label}
    if not (start_sec - 0.5) <= reference_sec <= (end_sec + 0.5):
        note = {
            "raw_line": line,
            "reason": "peak_outside_range_clamped",
            "label": label,
            "reference_sec": min(max(reference_sec, start_sec), end_sec),
        }
        reference_sec = min(max(reference_sec, start_sec), end_sec)
    event = AnnotatedEvent(
        demo_id=demo_id,
        label=label,
        start_sec=start_sec,
        end_sec=end_sec,
        reference_sec=reference_sec,
        raw_line=line,
    )
    return event, note


def _match_label(lowered: str) -> tuple[str | None, str]:
    for alias in sorted(LABEL_ALIASES, key=len, reverse=True):
        if lowered.startswith(alias):
            return LABEL_ALIASES[alias], lowered[len(alias) :]
    return None, lowered


def _match_excluded_label(lowered: str) -> str | None:
    for alias in sorted(EXCLUDED_ANNOTATION_LABELS, key=len, reverse=True):
        if lowered.startswith(alias):
            return EXCLUDED_ANNOTATION_LABELS[alias]
    return None


def build_event_classification_prompt(
    task_text: str,
    frame_groups: Sequence[tuple[str, Sequence[float]]],
    labels: Sequence[str] = EVENT_LABELS,
) -> str:
    """Build the forced-choice prompt for one selected peak.

    ``frame_groups`` pairs each camera with its frames' offsets from the peak, in the same
    order the images are attached, so the numbering in the prompt matches what the model
    receives and it knows which image is the moment being asked about.
    """

    frame_lines: list[str] = []
    image_number = 0
    for camera_name, offsets_sec in frame_groups:
        description = CAMERA_DESCRIPTIONS.get(camera_name, "camera")
        frame_lines.append(f'Camera "{camera_name}" ({description}), in chronological order:')
        for offset_sec in offsets_sec:
            image_number += 1
            if abs(offset_sec) < 1e-6:
                position = "THE QUERY MOMENT"
            elif offset_sec < 0:
                position = f"{abs(offset_sec):.2f}s before the query moment"
            else:
                position = f"{offset_sec:.2f}s after the query moment"
            frame_lines.append(f"  image {image_number}: {position}")
    option_lines = [f"  {label} --- {EVENT_LABEL_DESCRIPTIONS[label]}" for label in labels]
    camera_count = len(frame_groups)
    return (
        "You are analyzing a robot manipulation demonstration.\n"
        f'The robot is performing this task: "{task_text}"\n\n'
        f"You are shown {image_number} image(s) of the same moment from {camera_count} "
        "camera(s):\n"
        + "\n".join(frame_lines)
        + "\n\nClassify which event is happening at THE QUERY MOMENT. Choose exactly one "
        "of these options:\n"
        + "\n".join(option_lines)
        + "\n\nUse the neighbouring frames only to judge motion direction, and the wrist "
        "view only to judge contact with the object; classify the query moment itself. "
        "You must pick one option even if you are unsure.\n\n"
        'Return only a JSON object: {"event": "<one option, copied exactly>"}'
    )


def parse_event_prediction(raw_text: str, labels: Sequence[str] = EVENT_LABELS) -> str | None:
    """Extract a canonical label from the model's reply, or None if nothing matches."""

    if not raw_text:
        return None
    text = raw_text.strip()
    candidate = _label_from_json(text)
    if candidate is None:
        candidate = text
    lowered = candidate.strip().strip("\"'").lower()
    normalized = re.sub(r"[_\-]+", " ", lowered)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in LABEL_ALIASES:
        resolved = LABEL_ALIASES[normalized]
        return resolved if resolved in labels else None
    for label in sorted(labels, key=len, reverse=True):
        if label in normalized:
            return label
    for alias in sorted(LABEL_ALIASES, key=len, reverse=True):
        if alias in normalized and LABEL_ALIASES[alias] in labels:
            return LABEL_ALIASES[alias]
    return None


def _label_from_json(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("event", "label", "answer", "event_type"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def nearest_events(
    peak_sec: float,
    events: Sequence[AnnotatedEvent],
    tolerance_sec: float,
) -> tuple[list[AnnotatedEvent], float | None]:
    """Return every annotated event tied for the closest reference time to ``peak_sec``.

    Distance uses each event's reference time (the annotator's peak when given, the range
    midpoint otherwise). Several annotations frequently share one timestamp -- 13 of the
    ``gripper close`` labels sit on the same second as a ``first contact`` -- so returning
    the whole tie avoids making one of two simultaneous labels arbitrarily unreachable.
    Returns ``([], None)`` when nothing lies within ``tolerance_sec``.
    """

    if not events:
        return [], None
    deltas = [(abs(event.reference_sec - peak_sec), event) for event in events]
    best_delta = min(delta for delta, _ in deltas)
    if best_delta > tolerance_sec:
        return [], None
    tied = [event for delta, event in deltas if abs(delta - best_delta) <= TIE_EPSILON_SEC]
    return tied, best_delta


def aggregate_event_stats(
    rows: Sequence[dict[str, Any]],
    events_by_demo: dict[str, list[AnnotatedEvent]],
    tolerances_sec: Sequence[float] = DEFAULT_TOLERANCES_SEC,
    primary_tolerance_sec: float = 2.0,
    bootstrap_samples: int = 10000,
    seed: int = 20260823,
    labels: Sequence[str] = EVENT_LABELS,
) -> dict[str, Any]:
    """Score predictions per selector across a tolerance sweep.

    Reports the three quantities separately so a selector cannot hide behind either half
    of the task: ``match_rate`` is how often a selected frame lands near an annotated
    event at all, ``conditional_accuracy`` is how often the VLM labels those matched
    frames correctly, and ``end_to_end_accuracy`` is their product over all frames.
    """

    tolerance_values = sorted({float(value) for value in tolerances_sec} | {float(primary_tolerance_sec)})
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_selector[str(row["selector"])].append(row)

    stats: dict[str, Any] = {
        "labels": list(labels),
        "primary_tolerance_sec": float(primary_tolerance_sec),
        "tolerances_sec": tolerance_values,
        "bootstrap_samples": int(bootstrap_samples),
        "annotated_event_count": sum(len(events) for events in events_by_demo.values()),
        "annotated_demo_count": sum(1 for events in events_by_demo.values() if events),
        "ground_truth_label_counts": dict(
            Counter(event.label for events in events_by_demo.values() for event in events)
        ),
        "by_selector": {},
    }
    for selector, selector_rows in sorted(by_selector.items()):
        selector_stats: dict[str, Any] = {
            "frame_count": len(selector_rows),
            "demo_count": len({row["demo_id"] for row in selector_rows}),
            "unparsed_prediction_count": sum(
                1 for row in selector_rows if not row.get("predicted_label")
            ),
            "by_tolerance": {},
        }
        for tolerance in tolerance_values:
            selector_stats["by_tolerance"][f"{tolerance:g}"] = _score_at_tolerance(
                selector_rows,
                events_by_demo,
                tolerance_sec=tolerance,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                labels=labels,
            )
        selector_stats["primary"] = selector_stats["by_tolerance"][f"{primary_tolerance_sec:g}"]
        stats["by_selector"][selector] = selector_stats
    return stats


def _score_at_tolerance(
    rows: Sequence[dict[str, Any]],
    events_by_demo: dict[str, list[AnnotatedEvent]],
    *,
    tolerance_sec: float,
    bootstrap_samples: int,
    seed: int,
    labels: Sequence[str],
) -> dict[str, Any]:
    per_demo: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    hit_events: set[tuple[str, int]] = set()
    tied_frame_count = 0
    for row in rows:
        demo_id = str(row["demo_id"])
        events = events_by_demo.get(demo_id, [])
        matched, delta_sec = nearest_events(float(row["peak_sec"]), events, tolerance_sec)
        matched_labels = sorted({event.label for event in matched})
        predicted = row.get("predicted_label")
        is_matched = bool(matched)
        is_correct = bool(is_matched and predicted in set(matched_labels))
        per_demo[demo_id].append((is_matched, is_correct))
        if matched:
            if len(matched_labels) > 1:
                tied_frame_count += 1
            for event in matched:
                hit_events.add((demo_id, events.index(event)))
            # Credit the tied label the prediction actually matched so the diagonal stays
            # readable; misses are attributed to the first tied label.
            credited = predicted if is_correct else matched_labels[0]
            confusion[credited][str(predicted) if predicted else "unparsed"] += 1
        row[f"matched_labels@{tolerance_sec:g}"] = matched_labels or None
        row[f"matched_delta_sec@{tolerance_sec:g}"] = delta_sec
        row[f"correct@{tolerance_sec:g}"] = is_correct

    frame_count = sum(len(values) for values in per_demo.values())
    matched_count = sum(1 for values in per_demo.values() for matched, _ in values if matched)
    correct_count = sum(1 for values in per_demo.values() for _, correct in values if correct)
    evaluated_demos = {demo_id for demo_id in per_demo if events_by_demo.get(demo_id)}
    annotated_event_count = sum(len(events_by_demo.get(demo_id, [])) for demo_id in evaluated_demos)

    per_class = {}
    for label in labels:
        matched_for_label = sum(confusion[label].values())
        per_class[label] = {
            "matched_frame_count": matched_for_label,
            "correct_count": confusion[label].get(label, 0),
            "accuracy": _safe_ratio(confusion[label].get(label, 0), matched_for_label),
            "predictions": dict(confusion[label]),
        }
    majority_label = max(
        labels,
        key=lambda label: sum(confusion[label].values()),
        default=None,
    )
    majority_matched = sum(confusion[majority_label].values()) if majority_label else 0
    return {
        "tolerance_sec": float(tolerance_sec),
        "frame_count": frame_count,
        "matched_frame_count": matched_count,
        "tied_match_frame_count": tied_frame_count,
        "match_rate": _safe_ratio(matched_count, frame_count),
        "correct_count": correct_count,
        "conditional_accuracy": _safe_ratio(correct_count, matched_count),
        "end_to_end_accuracy": _safe_ratio(correct_count, frame_count),
        "annotated_event_count": annotated_event_count,
        "hit_event_count": len(hit_events),
        "event_recall_at_k": _safe_ratio(len(hit_events), annotated_event_count),
        "chance_accuracy": _safe_ratio(1, len(labels)),
        "majority_class_accuracy": _safe_ratio(majority_matched, matched_count),
        "majority_class_label": majority_label if majority_matched else None,
        "per_class": per_class,
        "clustered_bootstrap_95ci": _clustered_bootstrap(
            per_demo, bootstrap_samples=bootstrap_samples, seed=seed
        ),
    }


def _clustered_bootstrap(
    per_demo: dict[str, list[tuple[bool, bool]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample whole demos with replacement, mirroring the REASSEMBLE recall CIs."""

    demo_ids = sorted(per_demo)
    if not demo_ids or bootstrap_samples <= 0:
        return {}
    rng = random.Random(seed)
    match_rates: list[float] = []
    conditional: list[float] = []
    end_to_end: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [per_demo[rng.choice(demo_ids)] for _ in demo_ids]
        frames = sum(len(values) for values in sampled)
        matched = sum(1 for values in sampled for is_matched, _ in values if is_matched)
        correct = sum(1 for values in sampled for _, is_correct in values if is_correct)
        if frames:
            match_rates.append(matched / frames)
            end_to_end.append(correct / frames)
        if matched:
            conditional.append(correct / matched)
    return {
        "cluster_unit": "demo_id",
        "cluster_count": len(demo_ids),
        "match_rate": _percentile_interval(match_rates),
        "conditional_accuracy": _percentile_interval(conditional),
        "end_to_end_accuracy": _percentile_interval(end_to_end),
    }


def _percentile_interval(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "lower_2_5": _percentile(ordered, 2.5),
        "upper_97_5": _percentile(ordered, 97.5),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator
