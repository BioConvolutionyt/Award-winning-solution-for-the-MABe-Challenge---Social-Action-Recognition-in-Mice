verbose = True

import pandas as pd
import numpy as np
import itertools
import warnings
import json
import os, random
import gc
import lightgbm

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')

# Try importing additional models
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# Check GPU availability
GPU_AVAILABLE = False
try:
    import torch

    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
except:
    pass

# --- SEED EVERYTHING -----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)  # has to be set very early

rnd = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Cross-validation configuration
N_SPLITS = 5


class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples=None, max_imbalance_ratio=80.0):
        """
        estimator: Base learner
        n_samples: If None, no subsampling is performed; otherwise, stratified sampling is used to reduce to n_samples
        max_imbalance_ratio: Maximum majority-to-minority class ratio allowed before undersampling (e.g., 7.0 indicates 1:7)
        """
        self.estimator = estimator
        self.n_samples = n_samples  # if None → no subsampling/stratification
        self.max_imbalance_ratio = float(max_imbalance_ratio)

    def _to_numpy(self, X):
        try:
            return X.to_numpy(np.float32, copy=False)
        except AttributeError:
            return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        # First, process only the labels and indices; do not immediately convert the entire batch of X to numpy
        y = np.asarray(y).ravel()
        N = len(y)
        all_idx = np.arange(N)

        # Map {0,2} to {0,1}
        uniq = np.unique(y[~pd.isna(y)])
        if set(uniq.tolist()) == {0, 2}:
            y = (y > 0).astype(np.int8)

        # Perform majority class undersampling at the index level
        if self.n_samples is not None and self.max_imbalance_ratio > 0:
            n_samples = int(self.n_samples)
            if N > n_samples:
                valid_mask = ~pd.isna(y)
                y_valid = y[valid_mask]
                classes, counts = np.unique(y_valid, return_counts=True)
                if classes.size == 2:
                    min_idx_in_classes = int(np.argmin(counts))
                    maj_idx_in_classes = int(np.argmax(counts))
                    min_label = classes[min_idx_in_classes]
                    maj_label = classes[maj_idx_in_classes]
                    n_min = int(counts[min_idx_in_classes])
                    n_maj = int(counts[maj_idx_in_classes])

                    if n_min > 0 and n_maj > 0:
                        # Is the current ratio (minority:majority) less than 1:max_imbalance_ratio?
                        if self.max_imbalance_ratio * n_min < n_maj:
                            # To reach the 1:max_imbalance_ratio limit
                            maj_for_ratio = int(self.max_imbalance_ratio * n_min)
                            # To ensure the total sample count is not less than n_samples, the minimum majority class count
                            min_maj_allowed = max(n_samples - n_min, 0)
                            target_maj = min(n_maj, max(maj_for_ratio, min_maj_allowed))

                            if target_maj < n_maj and (n_min + target_maj) >= n_samples:
                                rng = np.random.RandomState(42)

                                maj_idx_all = np.where((y == maj_label) & valid_mask)[0]
                                min_idx_all = np.where((y == min_label) & valid_mask)[0]
                                maj_keep_idx = rng.choice(maj_idx_all, size=target_maj, replace=False)
                                nan_idx_all = np.where(~valid_mask)[0]

                                keep_idx = np.concatenate([min_idx_all, maj_keep_idx, nan_idx_all])
                                rng.shuffle(keep_idx)

                                # Update indices and labels (still no large Xn matrix created)
                                all_idx = keep_idx
                                y = y[all_idx]
                                N = len(y)

        # Perform stratified sampling on the subsampled index set to obtain n_samples
        if self.n_samples is not None and N > int(self.n_samples):
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.n_samples), random_state=42)
            try:
                dummy = np.zeros_like(y)
                sub_train_idx, _ = next(sss.split(dummy, y))  # sub_train_idx based on the current subset [0..N-1]
                train_idx = all_idx[sub_train_idx]            # Map back to the original indices
                y_train = y[sub_train_idx]
            except Exception as e:
                # Fallback: sample by step on the subset indices
                step = max(N // int(self.n_samples), 1)
                sub_train_idx = np.arange(0, N, step)
                train_idx = all_idx[sub_train_idx]
                y_train = y[sub_train_idx]
        else:
            train_idx = all_idx
            y_train = y

        if hasattr(X, "iloc"):
            X_sub = X.iloc[train_idx]
        else:
            X_sub = np.asarray(X)[train_idx]

        Xn = self._to_numpy(X_sub)
        self.estimator.fit(Xn, y_train)

        try:
            self.classes_ = np.asarray(self.estimator.classes_)
        except Exception:
            self.classes_ = np.unique(y_train)
        return self

    def predict_proba(self, X):
        Xn = self._to_numpy(X)
        try:
            P = self.estimator.predict_proba(Xn)
        except Exception:
            if len(self.classes_) == 1:
                n = len(Xn)
                c = int(self.classes_[0])
                if c == 1:
                    return np.column_stack([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
                else:
                    return np.column_stack([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)])
            return np.full((len(Xn), 2), 0.5, dtype=np.float32)

        P = np.asarray(P)
        if P.ndim == 1:
            P1 = P.astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        if P.shape[1] == 1 and len(self.classes_) == 2:
            P1 = P[:, 0].astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        return P

    def predict(self, X):
        Xn = self._to_numpy(X)
        try:
            return self.estimator.predict(Xn)
        except Exception:
            return np.argmax(self.predict_proba(Xn), axis=1)

# ==================== DATA LOADING ====================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft',
                   'headpiece_bottomfrontright',
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft',
                   'headpiece_topfrontright',
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']


def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():

        lab_id = row.lab_id
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid),
                                                      'frames')
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'],
                            annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(
                        vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        'frames_per_second': row.frames_per_second
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'],
                            annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions


# ==================== ADAPTIVE THRESHOLDING ====================

def finalize_features(df: pd.DataFrame, FEATURE_DTYPE=np.float32, SAFE_FEATURE_DOWNCAST=True) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if SAFE_FEATURE_DOWNCAST:
        # per-column safe downcast to avoid overflow; keep float64 if too large
        finfo = np.finfo(np.float32)
        for c in df.columns:
            col = df[c]
            if not np.issubdtype(col.dtype, np.floating):
                continue
            try:
                vmax = np.nanmax(np.abs(col.values))
            except Exception:
                vmax = None
            if vmax is not None and np.isfinite(vmax) and vmax <= finfo.max:
                try:
                    df[c] = col.astype(FEATURE_DTYPE, copy=False)
                except Exception:
                    pass
        return df
    try:
        return df.astype(FEATURE_DTYPE, copy=False)
    except Exception:
        return df

# ==================== ADVANCED FEATURE ENGINEERING ====================

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def _as_float64(series: pd.Series) -> np.ndarray:
    return np.asarray(series.values, dtype=np.float64)

#-----------------------------------------------------New-----------------------------------------------------

def add_long_range_temporal_features(X, data, part='body_center', fps=None, windows=[300]):
    if part not in data.columns:
        return X
    if 'x' not in data[part].columns or 'y' not in data[part].columns:
        return X

    for window in windows:
        ws = _scale(window, fps) if fps is not None else window
        ws = int(ws)
        scale = float(fps) if fps else 1.0

        sx = data[part]['x'].astype(float)
        sy = data[part]['y'].astype(float)
        sp_series = np.sqrt((sx - sx.shift(1)) ** 2 + (sy - sy.shift(1)) ** 2)

        sp = sp_series.to_numpy(np.float64, copy=False)
        if scale != 1.0:
            mask_valid = ~np.isnan(sp)
            sp[mask_valid] *= scale

        sp_s = pd.Series(sp, index=sp_series.index)
        trailing_max = sp_s.rolling(ws, min_periods=1).max()
        cnt = sp_s.rolling(ws, min_periods=1).count()
        trailing_max[cnt == 0] = -np.inf

        if ws <= 1:
            mean_center = sp_s.copy()
        else:
            w_center = ws + (ws % 2 == 0)
            mean_center = sp_s.rolling(w_center, min_periods=1, center=True).mean()

        mx = trailing_max.to_numpy(np.float64, copy=False)
        mn = mean_center.to_numpy(np.float64, copy=False)
        out = np.empty_like(mx)

        mn_valid = ~np.isnan(mn)
        out[~mn_valid] = 0.0
        out[mn_valid] = mx[mn_valid] / (mn[mn_valid] + 1e-6)

        X[f'{part}_activity_burst_{window}'] = out

    return X


def add_skip_gram_features(X, data, skip_distances=[30, 60, 90, 150], fps=None):
    available_parts = data.columns.get_level_values(0).unique()
    for part in ['tail_base']:
        if part in available_parts and 'x' in data[part].columns and 'y' in data[part].columns:
            x = _as_float64(data[part]['x'])
            y = _as_float64(data[part]['y'])
            n = x.shape[0]
            for skip in skip_distances:
                s = _scale(skip, fps) if fps is not None else skip
                s = abs(int(s))
                if s == 0:
                    out = np.zeros(n, dtype=np.float64)
                else:
                    out = np.empty(n, dtype=np.float64)
                    out[:s] = np.nan
                    dx = x[s:] - x[:-s]
                    dy = y[s:] - y[:-s]
                    out[s:] = np.sqrt(dx * dx + dy * dy)
                X[f'{part}_skip_dist_{skip}'] = out
    return X

#-------------------------------------------------------------------------------------------------------------

def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # invariant to time scaling

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    # displacement per frame is already in cm (pix normalized earlier); convert to cm/s
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features (windows & spans scaled by fps)."""
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM spans also interpreted in frames
    for span in [60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # per-frame velocities (cm/frame)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()  # decreasing distance => positive approach
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X

def add_facing_features(X, mouse_pair, fps):
    try:
        # require nose & tail_base for both
        if all(p in mouse_pair['A'].columns.get_level_values(0) for p in ['nose','tail_base']) and \
           all(p in mouse_pair['B'].columns.get_level_values(0) for p in ['nose','tail_base']):
            A_dir = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
            B_dir = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']

            # direction vectors normalized
            A_mag = np.sqrt(A_dir['x']**2 + A_dir['y']**2) + 1e-6
            B_mag = np.sqrt(B_dir['x']**2 + B_dir['y']**2) + 1e-6
            A_unit_x = A_dir['x'] / A_mag
            A_unit_y = A_dir['y'] / A_mag
            B_unit_x = B_dir['x'] / B_mag
            B_unit_y = B_dir['y'] / B_mag

            # vector from A to B
            ABx = (mouse_pair['B']['body_center']['x'] - mouse_pair['A']['body_center']['x'])
            ABy = (mouse_pair['B']['body_center']['y'] - mouse_pair['A']['body_center']['y'])
            AB_mag = np.sqrt(ABx**2 + ABy**2) + 1e-6

            # cos(angle between A facing dir and vector to B) -> 1 means A facing B
            X['A_face_B'] = (A_unit_x * (ABx/AB_mag) + A_unit_y * (ABy/AB_mag)).rolling(_scale(30,fps), min_periods=1, center=True).mean()
            # and symmetric
            BAx = -ABx; BAy = -ABy; BA_mag = AB_mag
            X['B_face_A'] = (B_unit_x * (BAx/BA_mag) + B_unit_y * (BAy/BA_mag)).rolling(_scale(30,fps), min_periods=1, center=True).mean()
    except Exception:
        pass
    return X


def transform_single(single_mouse, body_parts_tracked, fps):
    """Enhanced single mouse transform (FPS-aware windows/lags; distances in cm)."""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # Base distance features (squared distances across body parts)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Body angle (orientation)
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

        angle = np.arctan2(v1['y'], v1['x'])
        body_ang = np.arctan2(v2['y'], v2['x'])
        X['body_ang_diff'] = np.unwrap(angle - body_ang)  # unwrap reduces angle jumps
    
    # Core temporal features (windows scaled by fps)
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
            X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

        # Advanced features (fps-scaled)
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

        X = add_long_range_temporal_features(X, single_mouse, part='body_center', fps=fps)
        X = add_skip_gram_features(X, single_mouse, fps=fps)

        # NEW: Binary long distance features for 180 frames
        lag_180 = _scale(180, fps)
        if len(cx) >= lag_180:
            # Feature 1: Long-term displacement binary (has mouse moved far from position 180 frames ago?)
            long_disp = np.sqrt((cx - cx.shift(lag_180))**2 + (cy - cy.shift(lag_180))**2)
            X['longdist_bin1'] = (long_disp > 20.0).astype(float)  # Binary: moved >20cm in 180 frames
            
            # Feature 2: Sustained high activity binary (has activity been consistently high over 180 frames?)
            speed_180 = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
            X['longdist_bin2'] = (speed_180.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() > 5.0).astype(float)

    # Nose-tail features with duration-aware lags
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    # Ear features with duration-aware offsets
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)  
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return finalize_features(X)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Enhanced pair transform (FPS-aware windows/lags; distances in cm)."""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # Inter-mouse distances (squared distances across all part pairs)
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)
        shA = mouse_pair['A']['ear_left'].shift(lag)
        shB = mouse_pair['B']['ear_left'].shift(lag)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if '12+nose+tail_base' in X.columns and '12+ear_left+ear_right' in X.columns:
        X['elong'] = X['12+nose+tail_base'] / (X['12+ear_left+ear_right'] + 1e-6)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # Approach rate (duration-aware lag)
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # Temporal interaction features (fps-adjusted windows)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

        # NEW: Binary long distance features for 180 frames (pair interactions)
        lag_180 = _scale(180, fps)
        if len(cd_full) >= lag_180:    
            cd_dist = np.sqrt(cd_full)
            # Sustained far distance binary (have mice been consistently far apart for 180 frames?)
            X['longdist_pair_bin1'] = (cd_dist.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() > 30.0).astype(float)
            # Sustained close proximity binary (have mice been consistently close for 180 frames?)
            X['longdist_pair_bin2'] = (cd_dist.rolling(lag_180, min_periods=max(5, lag_180 // 6)).mean() < 10.0).astype(float)
    
    # Nose-nose dynamics (duration-aware lags)
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    # Velocity alignment (duration-aware offsets)
    if 'body_center' in avail_A and 'body_center' in avail_B:

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        # Advanced interaction (fps-adjusted internals)
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)
        X = add_facing_features(X, mouse_pair, fps)

        X = add_long_range_temporal_features(X, mouse_pair['A'], part='body_center', fps=fps)
        X = add_skip_gram_features(X, mouse_pair['A'], fps=fps)

    return finalize_features(X)


# ==================== ENSEMBLE TRAINING WITH GPU SUPPORT ====================


def build_base_models(n_samples):
    """Construct base models (LGBM / XGBoost / CatBoost) used for CV."""
    models = []

    # Configure GPU device for gradient boosting models
    gpu_device = 'gpu' if GPU_AVAILABLE else 'cpu'

    # LightGBM models
    models.append(
        make_pipeline(
            StratifiedSubsetClassifier(
                lightgbm.LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.07,
                    min_child_samples=40,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbose=-1,
                    device_type=gpu_device,
                    gpu_use_dp=True,
                    random_state=SEED,
                    bagging_seed=SEED,
                    feature_fraction_seed=SEED,
                    data_random_seed=SEED,
                ),
                int(n_samples / 1.3),
            )
        )
    )
    models.append(
        make_pipeline(
            StratifiedSubsetClassifier(
                lightgbm.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    min_child_samples=20,
                    num_leaves=63,
                    max_depth=9,
                    subsample=0.7,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    verbose=-1,
                    device_type=gpu_device,
                    gpu_use_dp=True,
                    random_state=SEED,
                    bagging_seed=SEED,
                    feature_fraction_seed=SEED,
                    data_random_seed=SEED,
                ),
                int(n_samples / 2),
            )
        )
    )

    # XGBoost model
    if XGBOOST_AVAILABLE:
        xgb_device = "gpu_hist" if GPU_AVAILABLE else "hist"
        models.append(
            make_pipeline(
                StratifiedSubsetClassifier(
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.08,
                        max_depth=7,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method=xgb_device,
                        verbosity=0,
                        random_state=SEED,
                    ),
                    int(n_samples / 1.5),
                )
            )
        )

    # CatBoost models
    if CATBOOST_AVAILABLE:
        cat_device = "GPU" if GPU_AVAILABLE else "CPU"
        models.append(
            make_pipeline(
                StratifiedSubsetClassifier(
                    CatBoostClassifier(
                        iterations=400,
                        learning_rate=0.07,
                        depth=7,
                        task_type=cat_device,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=SEED,
                    ),
                    n_samples,
                )
            )
        )
        models.append(
            make_pipeline(
                StratifiedSubsetClassifier(
                    CatBoostClassifier(
                        iterations=300,
                        learning_rate=0.1,
                        depth=9,
                        task_type=cat_device,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=SEED,
                    ),
                    n_samples,
                )
            )
        )

    return models


def cross_validate_action(X, y, groups, models, n_splits=N_SPLITS, min_pos=5):
    """
    Perform cross-validation using StratifiedGroupKFold on individual actions to generate out-of-forest (OOF) probabilities for each base model.
    Returns a numpy array with shape [n_samples, n_models].
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    n_samples = len(y)
    n_models = len(models)

    oof_probas = np.zeros((n_samples, n_models), dtype=np.float32)

    # Basic checks: number of groups / number of positive samples is sufficient
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits or (y == 0).all() or np.sum(y) < min_pos:
        return oof_probas

    cv = StratifiedGroupKFold(n_splits=n_splits)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y[tr_idx]

        for m_idx, base_model in enumerate(models):
            model = clone(base_model)
            try:
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_val)
                proba = np.asarray(proba)

                if proba.ndim == 1:
                    p1 = proba.astype(np.float32)
                elif proba.shape[1] == 1:
                    p1 = proba[:, 0].astype(np.float32)
                else:
                    p1 = proba[:, 1].astype(np.float32)

                oof_probas[val_idx, m_idx] = p1
            except Exception as e:
                if verbose:
                    print(f"    CV fold {fold}, model {m_idx} failed: {str(e)[:80]}")

        del X_tr, X_val, y_tr
        gc.collect()

    return oof_probas


def save_oof_parquet(section, switch, action, oof_probas, y):
    """
    Save the OOF probabilities and labels for a specific section / single|pair / action as a parquet file.
    Column names: proba_model0...proba_model{k}, label
    """
    y = np.asarray(y).astype(int)
    n_models = oof_probas.shape[1]

    data = {f"proba_model{i}": oof_probas[:, i] for i in range(n_models)}
    data["label"] = y
    df = pd.DataFrame(data)

    out_dir = os.path.join("oof_proba", str(section), str(switch))
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{action}.parquet")
    df.to_parquet(out_path, index=False)


# ==================== MAIN LOOP ====================

print(f"XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}\n")

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

        _fps_lookup = (
            train_subset[["video_id", "frames_per_second"]]
            .drop_duplicates("video_id")
            .set_index("video_id")["frames_per_second"]
            .to_dict()
        )

        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []

        for switch, data, meta, label in generate_mouse_data(train_subset, "train"):
            if switch == "single":
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            else:
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)

        # Single-mouse CV & OOF saving
        if len(single_list) > 0:
            single_feats_parts = []
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(Xi)

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
            single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
            single_meta = pd.concat(single_meta_list, axis=0, ignore_index=True)

            del single_list, single_label_list, single_meta_list, single_feats_parts
            gc.collect()

            print(f"  Single: {X_tr.shape}")

            models_single = build_base_models(n_samples=2_000_000)

            for action in single_label.columns:
                y_raw = single_label[action].to_numpy()
                action_mask = ~np.isnan(y_raw)

                y_action = y_raw[action_mask].astype(int)
                if y_action.size == 0:
                    continue

                X_action = X_tr[action_mask]
                groups_action = single_meta.loc[action_mask, "video_id"].to_numpy()

                oof_probas = cross_validate_action(X_action, y_action, groups_action, models_single, N_SPLITS)
                save_oof_parquet(section, "single", action, oof_probas, y_action)

                del X_action, y_action, groups_action, oof_probas
                gc.collect()

            del X_tr, single_label, single_meta, models_single
            gc.collect()

        # Mouse-pair CV & OOF saving
        if len(pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(Xi)

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
            pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
            pair_meta = pd.concat(pair_meta_list, axis=0, ignore_index=True)

            del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
            gc.collect()

            print(f"  Pair: {X_tr.shape}")

            models_pair = build_base_models(n_samples=900_000)

            for action in pair_label.columns:
                y_raw = pair_label[action].to_numpy()
                action_mask = ~np.isnan(y_raw)

                y_action = y_raw[action_mask].astype(int)
                if y_action.size == 0:
                    continue

                X_action = X_tr[action_mask]
                groups_action = pair_meta.loc[action_mask, "video_id"].to_numpy()

                oof_probas = cross_validate_action(X_action, y_action, groups_action, models_pair, N_SPLITS)
                save_oof_parquet(section, "pair", action, oof_probas, y_action)

                del X_action, y_action, groups_action, oof_probas
                gc.collect()

            del X_tr, pair_label, pair_meta, models_pair
            gc.collect()

    except Exception as e:
        print(f"***Exception*** {str(e)[:100]}")
        import traceback

        traceback.print_exc()
    gc.collect()
    print()