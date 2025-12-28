import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


OOF_BASE_DIR = "/kaggle/input/oof-proba-data/oof_proba"
OUTPUT_DIR = "merge_parameter"
RANDOM_STATE = 42
N_TRIALS = 500


def load_oof_data(parquet_path):
    """Load OOF probabilities and labels from a parquet file."""
    df = pd.read_parquet(parquet_path)
    if df is None or df.empty:
        return None, None
    if "label" not in df.columns:
        return None, None

    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None

    return X, y


def optimize_weights_threshold(X, y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    """Use TPE Bayesian optimization to find best blend weights and threshold."""
    n_models = X.shape[1]

    if n_models == 1:
        # Only one model, no need to optimize weights
        return [1.0], 0.5, f1_score(y, X[:, 0] >= 0.5, zero_division=0)

    def objective(trial):
        weights = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)], dtype=np.float64)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        threshold = trial.suggest_float("threshold", 0.0005, 0.49)

        blended = np.dot(X, weights)
        preds = (blended >= threshold).astype(int)
        score = f1_score(y, preds, zero_division=0)
        return score

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_weights = np.array(
        [study.best_params[f"w_{i}"] for i in range(n_models)],
        dtype=np.float64
    )
    if best_weights.sum() == 0:
        best_weights = np.ones_like(best_weights)
    best_weights = (best_weights / best_weights.sum()).tolist()

    best_threshold = float(study.best_params["threshold"])
    best_score = float(study.best_value)

    return best_weights, best_threshold, best_score



def save_parameters(section, switch, action, weights, threshold, score):
    out_dir = os.path.join(OUTPUT_DIR, section, switch)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{action}.json")
    data = {
        "weights": weights,
        "threshold": threshold,
        "best_f1": score,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_action(section, switch, parquet_path):
    action = os.path.splitext(os.path.basename(parquet_path))[0]

    try:
        X, y = load_oof_data(parquet_path)
    except Exception as e:
        print(f"Reading failed: {parquet_path} ({str(e)[:80]})")
        return

    if X is None or y is None:
        print(f"Skipping invalid file: {parquet_path}")
        return

    if np.unique(y).shape[0] <= 1:
        # Only one class; fallback to trivial weights/threshold
        weights = [1.0] + [0.0] * (X.shape[1] - 1)
        threshold = 0.5
        score = f1_score(y, (X[:, 0] >= threshold).astype(int), zero_division=0)
    else:
        try:
            weights, threshold, score = optimize_weights_threshold(X, y)
        except Exception as e:
            print(
                f"Optimization failed: section={section}, switch={switch}, action={action} ({str(e)[:80]}). "
                f"Using default mean fusion."
            )
            weights = [1.0 / X.shape[1]] * X.shape[1]
            threshold = 0.5
            blended = np.dot(X, np.array(weights))
            score = f1_score(y, (blended >= threshold).astype(int), zero_division=0)

    save_parameters(section, switch, action, weights, threshold, score)
    print(
        f"Section {section} [{switch}] Action {action}: "
        f"best F1={score:.4f}, threshold={threshold:.4f}, weights={weights}"
    )


def main():
    if not os.path.isdir(OOF_BASE_DIR):
        print(f"OOF directory does not exist: {OOF_BASE_DIR}")
        return

    sections = sorted(
        d for d in os.listdir(OOF_BASE_DIR)
        if os.path.isdir(os.path.join(OOF_BASE_DIR, d))
    )
    if not sections:
        print(f"No section subdirectories found in {OOF_BASE_DIR}.")
        return

    for section in sections:
        for switch in ["single", "pair"]:
            sec_dir = os.path.join(OOF_BASE_DIR, section, switch)
            if not os.path.isdir(sec_dir):
                continue

            parquet_paths = sorted(glob.glob(os.path.join(sec_dir, "*.parquet")))
            if not parquet_paths:
                continue

            for p_path in parquet_paths:
                process_action(section, switch, p_path)


if __name__ == "__main__":
    main()