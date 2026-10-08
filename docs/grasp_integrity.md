# Grasp-integrity metrics (bimanual, wrist cameras)

Module: `pathologist/metrics/grasp_integrity.py`. Detects slips and drops of an object held by either arm. Task-agnostic: it never reads task names, instructions or annotations.

## Signals

All four are right-tail ECOD sources (registered in `surprise.py`), sampled on the gripper telemetry grid, zero outside hold windows, and max-combined over the two arms.

| Metric | Source | Meaning |
|---|---|---|
| `grasp_attach_loss` | wrist camera | 1 − object/finger contact relative to contact at settle; a lost track counts as 1 |
| `grasp_object_gap` | wrist camera | shortest object-to-finger distance / image diagonal (direction-free) |
| `grasp_relative_motion` | wrist camera | object motion relative to the fingers, image diagonals per second |
| `grasp_aperture_collapse` | telemetry | measured aperture fall toward the command, divided by the possible fall |

## Method, per arm

1. **Hold windows (telemetry).** Close intent = commanded aperture below 0.4 (hysteresis exit at 0.6). Settle = the measured aperture stops moving. The window ends when the command rises 0.05 above its hold plateau. Each arm uses its own command; the two arms are never averaged.
2. **Fingers.** Segmented at the start of each window (Grounding DINO "robot gripper." boxes touching the image edge the fingers enter from, then SAM2), or a static mask from config. A thin contact band surrounds them.
3. **Held object.** SAM2 point prompts along the fingertip end of the band, plus generic text detections ("object. item. tool."). Candidates must touch the band. The chosen candidate is the one that **moves with the fingers** while the background moves with the arm (co-motion, measured on region interiors because dense flow blurs across motion boundaries); fingertip contact breaks ties and is the fallback when the arm is still.
4. **Tracking.** SAM2 point prompts from the chosen mask, through the window.
5. **Bimanual rules.** A drop is relabelled `transfer` (signals zeroed from then) if the other arm settles a grasp on an object within [−1.0 s, +0.5 s]. The other arm's mask is removed from candidates, and when it covers the contact band the frame is marked occluded (signals 0).
6. **Pulses, not plateaus.** After a confirmed drop, frames still in the lost state are zeroed after 0.5 s, so the persistence peak lands at the detachment rather than anywhere until the gripper opens.
7. **Pooling.** Frame values are max-pooled onto the telemetry grid, so 1–3 frame events survive alignment.

## Enabling it

Disabled by default (`metrics.grasp_integrity.enabled = false`). Example config:

```json
{
  "metrics": {
    "grasp_integrity": {
      "enabled": true,
      "empty_closed": {"left": 0.0, "right": 0.0},
      "vision": {"cache_dir": "grasp_cache", "debug_dir": null}
    }
  },
  "surprise": {
    "source_metrics": ["grasp_attach_loss", "grasp_object_gap",
                       "grasp_relative_motion", "grasp_aperture_collapse"]
  }
}
```

Wrist frames must be decoded: load ABC bundles with `camera_names` including `left_wrist` and `right_wrist`. Use at least 15 Hz (`camera_sample_hz`); drops last only a few native frames.

**Calibrate `empty_closed` once per arm:** the measured aperture when the gripper closes on nothing. It bounds how far the fingers can collapse; with the default 0.0, collapse is slightly underestimated if the real value is higher.

**ABC evaluation pipeline caveat (this branch):** `_abc_config_override` sets `surprise.source_metrics` after merging `--config`, so the grasp signals are computed there but cannot be added to its ECOD aggregate without changing that override.

## First check on real data

```bash
python scripts/check_grasp_integrity.py \
    --data-root /path/to/ABC-130k_subset/data/train \
    --output-dir grasp_integrity_check --max-episodes 10 --camera-sample-hz 15
```

Writes per-episode signal plots with failure segments shaded, a JSON summary of hold windows and events, and one initialization overlay per hold window under `overlays/` (fingers blue, band yellow, rejected candidates red, chosen object green). **Mark the overlays correct / wrong object / no object before using the metric in any evaluation**; if fewer than about 80% pick the held object, fix initialization first.

## Tests

`tests/test_grasp_integrity.py` (23 tests) covers hold windows (wide objects, hysteresis, open onset, per-arm independence, thin-object blindness), aperture collapse, contact band, candidate selection (co-motion, static-arm fallback), lost track and occlusion, max-pooling, the transfer rule, drop pulses, the window cache, engine/ECOD integration, and an end-to-end synthetic wrist-camera drop with a decoy object. Segmentation in the tests is a colour-based stand-in; **the Grounding DINO + SAM2 backend itself is not exercised by the tests** and needs the real-data check above.
