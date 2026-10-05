# TrajectoryPathologist: separate-tail aggregation

This branch contains the training-free PRISM event-selection pipeline used for
REASSEMBLE and ABC-130K (ABCBot). It converts robot telemetry, video, and
optional audio into trajectory-relative surprise curves, then selects salient
events using flood area or 0-D persistence.

Run all commands below from the repository root.

## Pipeline

1. A dataset adapter creates synchronized robot, camera, gripper, force, and
   audio streams when those modalities are available.
2. `MetricEngine` computes raw signals.
3. ECOD converts each source signal into trajectory-relative rarity scores.
4. Left-, right-, and skew-tail evidence is aggregated separately.
5. The final score is smoothed and converted into flood events.
6. Events are ranked by 0-D persistence.
7. The top K events are compared with interval annotations.

### Separate-tail aggregation

For metric `j` and timestamp `i`, ECOD produces `U_left[i,j]`,
`U_right[i,j]`, and `U_skew[i,j]`. With the default
`aggregation_reduction: "sum"`:

```text
A_left[i]  = sum_j U_left[i,j]
A_right[i] = sum_j U_right[i,j]
A_skew[i]  = sum_j U_skew[i,j]
score[i]   = max(A_left[i], A_right[i], A_skew[i])
```

`aggregation_reduction: "max"` replaces each cross-metric sum with a maximum.
Multi-component source metrics are first reduced with a per-tail component
maximum. Stalling metrics use the left tail; magnitude and event signals use
their configured right or skew tails.

## Features

The default ECOD aggregate uses:

- `action_discontinuity`
- `motion_smoothness`
- `velocity_spike`
- `stalling`
- `eef_linear_velocity_spike`
- `eef_angular_velocity_spike`
- `eef_translational_stalling`
- `eef_rotational_stalling`
- `eef_translational_jerk`
- `eef_angular_jerk`
- `gripper_aperture_speed`
- `visual_flow`
- `audio_spectral_flux`

Additional implemented diagnostics include timestamp regularity,
action-range saturation, action-distribution entropy, joint and Cartesian path
length, force norm, gripper transitions, and gripper chatter.

The evaluation stack also provides:

- REASSEMBLE HDF5 and ABC-130K MCAP adapters
- graceful omission of unavailable modalities
- reusable decoded-frame and robot-mask caches
- optional Grounding-DINO + SAM2 robot masking
- Uniform, KITE/flow, PRISM, and RONAR-K selectors
- area and 0-D-persistence ranking
- low-, high-, segment-, and trajectory-level evaluation
- trajectory-clustered bootstrap confidence intervals
- resumable runs
- aggregate plots, peak frames, and per-trajectory exports
- a Streamlit REASSEMBLE viewer
- optional VLM event classification

## Configuration and environment

The base configuration is `pathologist/default_config.json`. Pass a recursive
JSON override with `--config`.

Important settings:

```json
{
  "surprise": {
    "source_metrics": ["..."],
    "aggregation_reduction": "sum",
    "score_tail_overrides": {
      "stalling": "left"
    }
  },
  "selector": {
    "surprise_flood": {
      "smoothing_window": 10,
      "gamma": 0.5,
      "ranking": "persistence",
      "max_proposals": 5
    }
  }
}
```

Use the project Python environment:

```bash
conda activate trajectory-pathologist
python -m pytest -q
```

The pipeline requires NumPy, SciPy, h5py, OpenCV, matplotlib, MCAP protobuf
support, and compatible video codecs. Streamlit is required only for the
viewer. OpenAI credentials are required only for optional VLM classification.

Optional robot masking requires:

```bash
pip install -r requirements-robot-masking.txt
```

## REASSEMBLE

Pass one HDF5 trajectory with `--h5-path`, or a directory with `--data-root`.
The adapter reads the robot streams, `hand`, `hama1`, and `hama2` cameras, and
embedded high- and low-level demonstration annotations.

### Smoke test

```bash
python -m pathologist.run_reassemble_eval_pipeline \
  --data-root /path/to/REASSEMBLE \
  --output-dir /tmp/reassemble_smoke \
  --selectors metric \
  --metric-ranking persistence \
  --segment-level low \
  --max-files 1 \
  --max-demos 2 \
  --verbose
```

`--no-extract-frames` gives a telemetry-only debugging run. Do not use it when
evaluating visual flow or visual baselines.

### Failure localization

This evaluates failed high-level demonstrations against their failed
low-level intervals:

```bash
python -m pathologist.run_reassemble_eval_pipeline \
  --data-root /path/to/REASSEMBLE \
  --output-dir REASSEMBLE_failure_k5 \
  --top-k 5 \
  --selectors uniform kite_flow metric ronar_k \
  --segment-level low \
  --evaluation-unit failed_high_level_action \
  --metric-ranking persistence \
  --camera-name hama1 \
  --camera-name hama2 \
  --camera-name hand \
  --plot \
  --per-trajectory-results-dir REASSEMBLE_failure_k5/per_trajectory_results \
  --verbose
```

Use `--segment-level both` to score high- and low-level intervals. Resume an
interrupted run by repeating the same command with `--resume`.

### Localization viewer

```bash
python -m pathologist.run_reassemble_viewer_prep \
  --data-root /path/to/REASSEMBLE \
  --output-dir REASSEMBLE_viewer_k5 \
  --top-k 5 \
  --selectors uniform kite_flow metric \
  --segment-level low \
  --metric-ranking persistence \
  --evaluation-unit failed_high_level_action \
  --verbose

streamlit run pathologist/reassemble/viewer_app.py -- \
  --output-dir REASSEMBLE_viewer_k5
```

To overlay manually annotated events, add:

```bash
--annotations-path /path/to/annotations.txt \
--per-trajectory-results-dir REASSEMBLE_localization/per_trajectory_results
```

`--annotation-label` may be repeated to select event classes.

### Successful-trajectory event localization

```bash
python -m pathologist.run_reassemble_eval_pipeline \
  --data-root /path/to/REASSEMBLE \
  --output-dir REASSEMBLE_success_k5 \
  --top-k 5 \
  --selectors uniform kite_flow metric \
  --segment-level low \
  --evaluation-unit successful_high_level_action \
  --metric-ranking persistence \
  --plot \
  --annotations-path /path/to/annotations.txt \
  --per-trajectory-results-dir REASSEMBLE_success_k5/per_trajectory_results \
  --verbose
```

Optional VLM event classification and tolerance-based scoring:

```bash
export OPENAI_API_KEY=...

python -m pathologist.run_reassemble_event_qa_pipeline \
  --peaks-dir REASSEMBLE_success_k5 \
  --annotations /path/to/annotations.txt \
  --output-dir REASSEMBLE_success_event_qa \
  --selectors uniform kite_flow metric \
  --cameras hama1 hama2 hand \
  --context-frames 1 \
  --context-offset-sec 0.4 \
  --tolerance-sec 2 \
  --bootstrap-samples 10000 \
  --resume \
  --verbose
```

Use `--skip-vlm` to inspect prompts without API calls and `--score-only` to
rescore existing predictions.

### REASSEMBLE outputs

- `reassemble_manifest.json`: inclusion, coverage, and metric availability
- `reassemble_eval.jsonl`: per-trajectory selector results
- `reassemble_stats.json`: aggregate precision and recall
- `reassemble_alignment_debug.json`: annotation alignment diagnostics
- `run_summary.json`: run settings and output paths
- `selected_peaks/`: selected peak records and frames
- `frame_cache/`: reusable decoded frames
- `ecod_aggregates/`: aggregate curves when `--plot` is enabled

## ABC-130K / ABCBot

Expected layout:

```text
<data-root>/
  <task-name>/
    <episode-id>/
      episode.mcap
      annotation.mcap
```

An exact, case-insensitive `mistake` annotation defines a failure interval
that ends at the next annotation. By default the evaluator selects one failure
episode per task. Use `--all-failure-trajectories` for the complete downloaded
failure cohort.

### Smoke test

```bash
python -m pathologist.run_abc130k_eval_pipeline \
  --data-root /path/to/ABC-130k/data/train \
  --output-dir /tmp/abc_smoke \
  --top-k 5 \
  --selectors metric \
  --metric-ranking persistence \
  --smoothing-window 10 \
  --max-episodes 2 \
  --bootstrap-samples 0 \
  --verbose
```

### Full evaluation

```bash
python -m pathologist.run_abc130k_eval_pipeline \
  --data-root /path/to/ABC-130k/data/train \
  --output-dir ABC130K_k5 \
  --top-k 5 \
  --selectors uniform kite_flow metric ronar_k \
  --all-failure-trajectories \
  --metric-ranking persistence \
  --smoothing-window 10 \
  --camera-sample-hz 5 \
  --bootstrap-samples 10000 \
  --seed 0 \
  --verbose
```

For comparable ablations, reuse frames:

```bash
--frame-cache-dir /path/to/shared/abc_frame_cache
```

Resume by repeating the same command with `--resume`. Optional robot-masked
visual flow is enabled with:

```bash
--robot-mask-visual-flow \
--robot-mask-cache-dir /path/to/shared/robot_mask_cache
```

### ABCBot outputs

- `abc130k_manifest.json`: episode status and metric availability
- `abc130k_eval.jsonl`: selected peaks and per-episode scores
- `abc130k_stats.json`: aggregate and paired bootstrap statistics
- `run_summary.json`: locked run settings
- `frame_cache/`: reusable decoded frames
- `robot_mask_cache/`: optional cached masks

## Evaluation definitions

- **Point Precision@K:** selected peaks inside a target interval divided by
  selected peaks.
- **Segment Recall@K:** target intervals containing at least one peak divided
  by target intervals.
- **Trajectory/Episode Recall@K:** trajectories with at least one target hit
  divided by evaluated trajectories.

Point precision measures purity. Segment recall measures coverage of distinct
events and does not reward repeated peaks in one interval.

For reproducibility:

- use a new output directory for each configuration;
- resume only with identical settings;
- reuse frame and mask caches;
- inspect manifest metric availability instead of treating missing metrics as
  zeros;
- report K, annotation level, ranking, smoothing, camera rate, and padding;
- report ABC EEF availability for EEF-focused analyses.
