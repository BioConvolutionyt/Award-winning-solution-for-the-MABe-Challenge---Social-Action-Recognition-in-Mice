# Award-winning solution for the MABe Challenge - Social Action Recognition in Mice

This project is a distilled core version of a reproducible solution for the Kaggle competition **MABe Mouse Behavior Detection**. It keeps the best-performing feature engineering and the **LGBM + XGB + CatBoost** ensemble strategy, and uses **OOF probabilities + TPE** to automatically optimize per-action blending weights and binary classification thresholds. Finally, it trains on the full training set and generates `submission.csv`.

- Competition link: [MABe Mouse Behavior Detection](https://www.kaggle.com/competitions/mabe-mouse-behavior-detection)

---

## Project Structure

Only **3 core scripts** are kept in this directory:

- `OOF probability.py`
  - **Purpose**: For each `section (body_parts_tracked)`, `single/pair`, and `action`, train a binary classifier and run **5-fold StratifiedGroupKFold (grouped by video_id)** to produce OOF probabilities for each base model.
  - **Output**: `oof_proba/{section}/{single|pair}/{action}.parquet`

- `Opt merge parameters.py`
  - **Purpose**: Read OOF probabilities and use **Optuna + TPE** to search the **best blending weights** and **best binary threshold** for each action.
  - **Output**: `merge_parameter/{section}/{single|pair}/{action}.json`

- `MABe++ Refined LGB+XGB+Catboost.py`
  - **Purpose**: Train with the same model configuration as OOF on the full training set; during inference, load the per-action optimal weights/thresholds from `merge_parameter`, apply temporal smoothing and event-level post-processing, and generate `submission.csv`.

---

## End-to-End Workflow (Recommended: Kaggle Notebook)

The 3 steps must be executed in this order:

### 1) Generate OOF probabilities

Run `OOF probability.py`:

- **Inputs** (hard-coded to Kaggle input paths in the script):
  - `/kaggle/input/MABe-mouse-behavior-detection/train.csv`
  - `/kaggle/input/MABe-mouse-behavior-detection/test.csv`
  - `/kaggle/input/MABe-mouse-behavior-detection/train_tracking/*/*.parquet`
  - `/kaggle/input/MABe-mouse-behavior-detection/train_annotation/*/*.parquet`
- **Outputs**:
  - `oof_proba/{section}/{single|pair}/{action}.parquet`

It is recommended to package `oof_proba/` as a Kaggle Dataset (e.g., named `oof-proba-data`) for mounting in the next step.

### 2) TPE optimization for blending weights and thresholds

Run `Opt merge parameters.py`:

- **Input**: `OOF_BASE_DIR` points to the `oof_proba` folder in a Kaggle Dataset (default in the script):
  - `OOF_BASE_DIR = "/kaggle/input/oof-proba-data/oof_proba"`
- **Outputs**:
  - `merge_parameter/{section}/{single|pair}/{action}.json`

It is also recommended to package `merge_parameter/` as a Kaggle Dataset (e.g., named `opt-parameters`) for mounting in the next step.

### 3) Full training + inference + submission generation

Run `MABe++ Refined LGB+XGB+Catboost.py`:

- **Inputs**:
  - MABe raw data: same as Step 1
  - Optimized parameters: loaded from a Kaggle Dataset by default
    - `MERGE_PARAM_DIR = "/kaggle/input/opt-parameters/merge_parameter"`
- **Output**:
  - `submission.csv`

---

## Data and Task Definition (Brief)

### Data format

- Each video corresponds to a time-series tracking file (`train_tracking/*.parquet`), containing x/y coordinates of multiple body keypoints for multiple mice.
- `train.csv` provides video-level metadata (e.g., `video_id`, `frames_per_second`, `pix_per_cm_approx`, `body_parts_tracked`, etc.) and which behavior labels are available for that video (`behaviors_labeled`).
- `train_annotation/*.parquet` contains behavior event intervals (start/stop frames). The scripts expand them into per-frame 0/1 label sequences.

### Prediction target (submission format)

The final submission is event-interval based: `video_id, agent_id, target_id, action, start_frame, stop_frame`.  
During inference, the pipeline first produces per-frame action probabilities, then segments them into event intervals through thresholds and post-processing.

---

## Core Approach: Per-Action Binary Modeling + OOF + Automatic Optimization

Overall, this is a pipeline of “use OOF for blending/threshold calibration first, then train on all data for inference”:

```mermaid
flowchart TD
  rawData[RawData(train_tracking/train_annotation/train.csv)] --> oof[OOF_probability_py]
  oof --> oofFiles[oof_proba_section_switch_action_parquet]
  oofFiles --> opt[Opt_merge_parameters_py]
  opt --> params[merge_parameter_section_switch_action_json]
  params --> infer[MABepp_Refined_LGB_XGB_Catboost_py]
  rawData --> infer
  infer --> sub[submission_csv]
```

### 0) Task decomposition: section / single|pair / action

All three scripts share the same decomposition granularity:

- **section**: Different values of `train.body_parts_tracked` (enumerated by `body_parts_tracked_list` and iterated by index). Each section represents “which body parts exist in the tracking for that video”.  
  - When there are too many body parts, some are filtered via `drop_body_parts` (e.g., head-mounted devices, spine, mid-tail points, etc.).
- **switch**: `single` (self behavior) vs `pair` (social/interaction behavior).
- **action**: Each specific behavior (e.g., `rear`). This solution treats each action as an **independent binary classification** task for training and calibration.

### 1) Sample generation and label construction (by video_id, by agent/target)

Both `OOF probability.py` and `MABe++ Refined LGB+XGB+Catboost.py` use `generate_mouse_data(...)` to expand one video into multiple training samples (frame-level time-series samples):

- **Tracking read**: Load keypoint coordinates from `{train|test}_tracking/{lab_id}/{video_id}.parquet`, and pivot into a table indexed by `video_frame` with columns `(mouse_id, bodypart, coord[x|y])`.
- **Unit normalization**: Use `pix_per_cm_approx` to normalize scale (convert pixels to approximate centimeters), so different videos become comparable.
- **single samples**:
  - Only keep annotations with `target == 'self'`;
  - For each `agent (mouse_id)`, generate one sample: features are derived from that mouse’s trajectory;
  - Labels come from `train_annotation` intervals where `(agent_id==mouse_id, target_id==mouse_id)`, expanded into per-frame 0/1; unannotated frames can be `NaN` (masked later).
- **pair samples**:
  - Only keep annotations with `target != 'self'`;
  - For each ordered pair `(agent, target)` generated by `itertools.permutations(..., 2)`, create one sample; features concatenate both mice’s trajectories (marked as A/B);
  - Labels come from `train_annotation` intervals where `(agent_id==agent, target_id==target)`, expanded into per-frame 0/1.

> Key point: training/OOF/inference all work on **frame-level probabilities**, and the final output is event-intervals after segmentation.

### 2) Feature engineering (FPS-aware)

Two input types (`single` / `pair`) go through `transform_single(...)` and `transform_pair(...)`. Feature categories include:

- **Geometry / morphology**: pairwise body-part distances (squared distances), elongation, etc.
- **Velocity and activity**: displacement/velocity statistics, multi-window rolling mean/std.
- **Pose / angles**: body angles and their changes.
- **Multi-scale temporal statistics**: means/variances/ratios under multi-scale windows (scaled by FPS).
- **Long-range features**: long-window activity bursts, skip-gram distances, etc.
- **Interaction features (pair)**: inter-mouse distance, chasing/approach, speed correlation, facing/orientation, etc.
- **Numerical robustness**: `finalize_features` handles `inf/-inf` and tends to cast to `float32` to reduce memory.

Window lengths are scaled by `_scale(frames_at_30fps, fps)` using the video FPS, avoiding temporal scale mismatch caused by assuming a fixed 30 FPS.

### 3) Stratified group cross-validation (avoid video leakage)

In `OOF probability.py`, for each action (binary label), we use:

- **CV**: `StratifiedGroupKFold(n_splits=5, groups=video_id)`

This ensures the **same video never appears in both train and validation folds**, reducing leakage common in time-series problems (frames from the same source video split across folds).

The script also includes basic skipping rules:

- **Minimum positives**: `min_pos=5` (if too few positives, return all-zero OOF)
- **Group count check**: `unique(video_id) >= n_splits`
- **All-zero labels**: if an action is all zeros for that block, skip/return all-zero

### 4) Base models: 2×LGBM + 1×XGB + 2×CatBoost (fixed configuration)

`build_base_models(...)`/`submit_ensemble(...)` defines 5 base models (fixed params):

- **LightGBM** (deep)
- **LightGBM** (very deep)
- **XGBoost**
- **CatBoost** (deep)
- **CatBoost** (very deep)

If the environment does not have xgboost/catboost installed, they are skipped automatically.

GPU auto-detection and configuration:

- **LightGBM**: `device_type='gpu'` and `gpu_use_dp=True`
- **XGBoost**: `tree_method='gpu_hist'`
- **CatBoost**: `task_type='GPU'`

If no GPU is available, the corresponding CPU configs are used.

### 5) Sampling and class-imbalance handling for large-scale training (StratifiedSubsetClassifier)

Because frame-level samples can be huge, each base learner executes inside `fit`:

- **Stratified subsampling**: when the dataset is too large, only train on `n_samples`. In the scripts:
  - single: `n_samples=2_000_000`
  - pair: `n_samples=900_000`
  - and different models may scale this (e.g., `n_samples/1.3`, `n_samples/2`, `n_samples/1.5`, etc.).
- **Majority-class undersampling**: when the binary task is extremely imbalanced, downsample the majority class to not exceed `max_imbalance_ratio` (default `80.0`).

Implementation prioritizes sampling at the “indices and y label” level, and only converts the selected `X` subset to numpy at the end, reducing peak memory usage.

### 6) OOF probability outputs (for downstream blending/threshold calibration)

For each `section / single|pair / action`, one OOF parquet is generated:

- **Path**: `oof_proba/{section}/{single|pair}/{action}.parquet`
- **Columns**:
  - `proba_model0 ... proba_model{k}`: OOF probability for the positive class from each base model (k depends on which models are available)
  - `label`: 0/1 ground-truth label

### 7) TPE optimization: blending weights + binary threshold (optimize per action independently)

`Opt merge parameters.py` reads the parquet files and runs an Optuna-TPE optimization per action:

- **Trials**: `N_TRIALS = 500`
- **Search space**:
  - weights: `w_i ~ Uniform(0, 1)`, then normalized so `sum(w)=1`
  - threshold: `threshold ~ Uniform(0.0005, 0.49)`
- **Objective**: maximize `f1_score(y, (X @ w) >= threshold)` and use `zero_division=0` to avoid errors when no positives are predicted
- **Single-class fallback**: if `y` has only one class in a parquet, fall back to simple default weights/threshold and compute the score directly
- **Output**: `merge_parameter/{section}/{single|pair}/{action}.json`
  - `weights` (list[float])
  - `threshold` (float)
  - `best_f1` (float)

### 8) Inference: blending, smoothing, multi-action decision, and event output

Key inference logic in `MABe++ Refined LGB+XGB+Catboost.py`:

- **Per-action frame-level probabilities**:
  - For each test sample, compute `X_te`
  - For each action: base models produce probabilities, stacked into `stacked`, then blended by `weights` from `merge_parameter` to get `pred[action]`
  - If optimization parameters are missing for an action, fall back to default weights (`default_weights = [0.20, 0.15, 0.25, 0.22, 0.18]` in the script; truncated and re-normalized to match the available model count)

- **Temporal smoothing (aligned with the pipeline)**:
  - In `predict_multiclass_adaptive`, apply `rolling(window=12, min_periods=1, center=True).mean()` to `pred`

- **From multi-action probabilities to “one action / other per frame”**:
  - Take `argmax` per frame to get a candidate action
  - Apply the action-specific threshold (separate tables for single/pair), and thresholds are scaled by:
    - `effective_threshold = threshold * THRESHOLD_SCALE`
  - If the threshold is not met, label as `other` (internally marked as `-1`).

- **Event segmentation and filtering**:
  - Segment by “whether the action changes between adjacent frames”
  - Filter out very short events (`duration >= 3` in the script)

- **robustify constraints (ensure the submission is valid)**:
  - Remove invalid rows (`start_frame < stop_frame`, drop NaN) and cast to int
  - Group by `(video_id, agent_id, target_id)` and avoid overlapping intervals
  - Fill uncovered intervals with `other` (batch concatenation) to ensure the full timeline is covered

---

## Directory Conventions and Artifacts

- **OOF probabilities**: `oof_proba/{section}/{single|pair}/{action}.parquet`
- **Blending parameters**: `merge_parameter/{section}/{single|pair}/{action}.json`
- **Submission**: `submission.csv`

---

## FAQ

### 1) Why GroupKFold / StratifiedGroupKFold?

Frames within the same `video_id` are highly correlated. If you randomly shuffle frames, the training and validation sets can share fragments of the same video, causing overly optimistic validation scores and distorted parameter optimization. Splitting by `video_id` significantly reduces leakage risk.

### 2) What is `THRESHOLD_SCALE` and why does it matter?

- **Evaluation peculiarity**: The competition evaluates predictions based on the “declared labeled behavior set” in the ground truth (corresponding to `behaviors_labeled` in the code). In other words, some labs/videos only label a subset of actions (e.g., only `rear`), or only label certain mice/pairs. For action/pair combinations that are **not declared / not labeled**, positive predictions are typically not counted as false positives (effectively ignored). This means the data naturally contains “other actions happen but are not labeled” missing-label situations, consistently in both train and test.
- **Why scale thresholds**: Under this “partial labeling / intersection-style evaluation”, making predictions a bit more aggressive (lower thresholds, higher recall) is often beneficial: some extra positives fall into the non-evaluated label set (no penalty), while recall gains on evaluated actions can yield net improvements.
- **How it is applied**: Use a global scaling factor to shift all action thresholds:
  - `effective_threshold = threshold * THRESHOLD_SCALE` (more aggressive when `THRESHOLD_SCALE < 1`)
- **Empirical effect**: Even after per-action threshold optimization, adding `THRESHOLD_SCALE` can still yield around `0.001 ~ 0.003` score improvement in practice (a heuristic; tune it based on validation/online feedback).

### 3) How to run locally?

The scripts use Kaggle paths (`/kaggle/input/...`) by default. To run locally, you need to modify those paths in the code to your local data directory (this project follows the constraint “all configurable items live in the code”, no command-line arguments).

---

## Dependencies

Main dependencies:

- `pandas`, `numpy`, `scikit-learn`
- `lightgbm`
- `xgboost`
- `catboost`
- `optuna` (for TPE optimization)
- `polars` (used for scoring / post-processing utilities)


