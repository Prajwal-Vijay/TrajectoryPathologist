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
2. **Fingers (calibrated, not detected).** The wrist camera is bolted to the gripper, so the fingers' pixels depend only on the aperture and are the same in every episode. Everything else changes between episodes. Calibration therefore bins frames from many episodes by measured aperture and keeps, per bin, the pixels whose grey level barely varies (per-pixel std below an Otsu threshold clipped to [6, 30]), connected to the image edge the fingers enter from, holes filled. Hold windows (±0.5 s) are left out, because a held object is rigid with the camera too. A window uses the mask of the bin nearest its measured hold aperture. Text-prompted detection ("robot gripper.") is only used if `finger_fallback: "segmented"` is set: in wrist views its box contains whatever the fingers hold, so SAM2 returned the held object, the other arm or most of the image as "fingers".
3. **Held object (rigidity).** Candidates are SAM2 masks from points along the fingertip end of a thin band around the fingers, plus generic text detections; they must touch the band. The held object is rigid with the camera, so while the arm moves it keeps its pixels and appearance and the background does not. For each candidate, rigidity = median over frames in which the background has visibly changed (≥ 15% of textured scene pixels differ from the settle frame by more than 10 grey levels; up to 6 such frames, taken from anywhere in the window because the arm is usually still right after the grasp) of the share of its *evidence pixels* still within 10 grey levels of the settle frame. Evidence pixels are textured interior pixels (local 7×7 std ≥ 12) and an inner ring along the candidate's boundary away from the fingers and image border. A plain surface carries no evidence in its interior: it looks identical after the camera pans, and optical flow there is zero. That is why the previous flow-based "moves with the fingers" test picked plain walls. The chosen candidate has the highest rigidity (≥ 0.6); fingertip contact breaks ties within 0.1. If the background never visibly changes during the hold, confidence is `low` and the candidate with most fingertip contact is used.
4. **Tracking.** SAM2 point prompts from the chosen mask, through the window.
5. **Bimanual rules.** A drop is relabelled `transfer` (signals zeroed from then) if the other arm settles a grasp on an object within [−1.0 s, +0.5 s]. The other arm's mask is removed from candidates, and when it covers the contact band the frame is marked occluded (signals 0).
6. **Pulses, not plateaus.** After a confirmed drop, frames still in the lost state are zeroed after 0.5 s, so the persistence peak lands at the detachment rather than anywhere until the gripper opens.
7. **Pooling.** Frame values are max-pooled onto the telemetry grid, so 1–3 frame events survive alignment.

## Enabling it

Disabled by default (`metrics.grasp_integrity.enabled = false`). Vision needs a finger calibration per wrist camera (built by the check script below); without one, windows get `vision_status = "no_finger_calibration"` and only the telemetry signal is produced. Example config:

```json
{
  "metrics": {
    "grasp_integrity": {
      "enabled": true,
      "empty_closed": {"left": 0.0, "right": 0.0},
      "vision": {
        "cache_dir": "grasp_cache",
        "finger_calibration": {
          "left_wrist": "grasp_integrity_check/finger_calibration/left_wrist.npz",
          "right_wrist": "grasp_integrity_check/finger_calibration/right_wrist.npz"
        }
      }
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

First it calibrates the fingers from `--calib-episodes` (default 60) episodes spread across tasks, at 2 Hz, and writes `finger_calibration/<camera>.npz` and a PNG with one mask per aperture bin. **Look at that PNG first:** each panel should contain the fingers and nothing else. If a bin includes a table edge or an object, use more episodes or raise `--calib-min-episodes`. The calibration is reused on later runs unless `--recalibrate` is given. If the gripper is open whenever it is empty, there is only an "open" bin, and grasps use it: the fingertip pads then sit a little inside the band, which shifts the signals by a constant during the hold but does not hide a drop (ECOD is rank-based within the trajectory).

Then, per episode with an annotated failure, it writes signal plots with failure segments shaded, a JSON summary of hold windows and events, and one two-panel overlay per hold window under `overlays/`. Left: the settle frame with fingers (blue), contact band (yellow), candidates that touched the band but were not chosen (red outline), the chosen object (green outline) and each candidate's rigidity `r`. Right: a later frame used to judge rigidity, with the settle-frame outlines; when the choice is right the object still fills the green outline there while the background has moved. File names end in the confidence (`high`, `low`, `none`). The printed summary counts windows by status and confidence. **Mark the overlays correct / wrong object / no object before using the metric in any evaluation**; if fewer than about 80% of `high` windows pick the held object, fix initialization first.

Known limits: automatic exposure changes as the arm turns can push a held object's pixels past the 10-grey-level tolerance and lower its rigidity; a plain background next to another plain region of the same grey level shows no boundary motion.

## Tests

`tests/test_grasp_integrity.py` (32 tests) covers hold windows (wide objects, hysteresis, open onset, per-arm independence, thin-object blindness), aperture collapse, contact band, finger calibration (aperture bins, nearest bin, exclusion of held objects, round trip), candidate selection (rigidity vs a moving table, a plain moving wall that has zero optical flow, candidates without evidence, evidence arriving only once the arm moves, static-arm fallback), lost track and occlusion, max-pooling, the transfer rule, drop pulses, the window cache, engine/ECOD integration, and an end-to-end synthetic wrist-camera drop with a decoy object. Segmentation in the tests is a colour-based stand-in and the scenes are synthetic; **SAM2 candidates, real exposure changes and real backgrounds are not exercised by the tests** and need the real-data check above.
