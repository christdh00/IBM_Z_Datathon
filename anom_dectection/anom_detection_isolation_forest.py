import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.signal import periodogram

# Config
DATA_FOLDER = "../PW_0.5h"
TARGET = DATA_FOLDER
TRAIN_PATH = f"{DATA_FOLDER}/train.csv"
VAL_PATH = f"{DATA_FOLDER}/val.csv"
TEST_PATH = f"{DATA_FOLDER}/test.csv"

SENSORS = ["CMS air pressure", "Concentrazione ossigeno base", "Nitrogen pressure", "Oxygen over threshold"]

#helper functions
def spectral_entropy(x):
    """Return spectral entropy, or 0 when there is no usable spectral power."""
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        return 0.0

    _, psd = periodogram(x, scaling="spectrum")
    total_power = np.sum(psd)

    # A constant signal has no spectral power after periodogram detrending.
    if not np.isfinite(total_power) or total_power <= 0:
        return 0.0

    psd_norm = psd / total_power
    return entropy(psd_norm)

def derivative_features(x):
    """Compute derivative features; short signals have no measurable changes."""
    dx = np.diff(x)

    if dx.size == 0:
        return {
            "mean_derivative": 0.0,
            "max_derivative": 0.0,
            "num_sign_changes": 0,
            "flat_segments": 0,
            "longest_flat_segment": 0,
        }

    # Ignore zero derivatives when counting direction reversals.
    nonzero_dx = dx[dx != 0]

    return {
        "mean_derivative": np.mean(np.abs(dx)),
        "max_derivative": np.max(np.abs(dx)),
        "num_sign_changes": np.sum(np.diff(np.sign(nonzero_dx)) != 0),
        "flat_segments": np.sum(dx == 0),
        "longest_flat_segment": longest_run(dx == 0),
    }

def longest_run(arr):
    """Longest consecutive True values."""
    max_run = run = 0
    for val in arr:
        if val:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def out_of_range_fraction(x, min_val, max_val):
    return np.mean((x < min_val) | (x > max_val))

def extreme_jump_fraction(x, threshold):
    dx = np.abs(np.diff(x))

    # A signal with fewer than two readings has no jumps to count.
    if dx.size == 0:
        return 0.0

    return np.mean(dx > threshold)

#Feature engineering
def extract_features(df, ranges, jump_thresholds, baseline_stats=None):
    """
    df: RW dataframe (rows = timestamps, columns = sensors)
    ranges: dict {sensor: (min_val, max_val)}
    jump_thresholds: dict {sensor: max_allowed_jump}
    baseline_stats: dict {sensor: (mean, std)} for drift detection
    """

    if df.empty:
        raise ValueError("Cannot extract features from an empty RW.")
    
    features = {}
    
    # -----------------------------------------------------
    # RW-level missingness indicators
    # -----------------------------------------------------
    missing_fraction_per_sensor = df.isna().mean()
    features["rw_missing_fraction"] = missing_fraction_per_sensor.mean()
    features["rw_num_high_missing_sensors"] = (missing_fraction_per_sensor > 0.2).sum()
    
    # -----------------------------------------------------
    # 1. Per-sensor features
    # -----------------------------------------------------
    for sensor in df.columns:
        x_raw = pd.to_numeric(df[sensor], errors="raise")
        x_raw = x_raw.replace([np.inf, -np.inf], np.nan)

        all_missing = x_raw.isna().all()

        if all_missing:
            # No readings exist to compute signal statistics from.
            # Missingness features below will still record this clearly.
            x = np.zeros(len(x_raw), dtype=float)
        else:
            x = x_raw.ffill().bfill().to_numpy(dtype=float)
        
        # Missingness features
        features[f"{sensor}_missing_fraction"] = x_raw.isna().mean()
        features[f"{sensor}_missing_count"] = x_raw.isna().sum()

        streaks = x_raw.isna().astype(int)
        features[f"{sensor}_missing_longest_streak"] = longest_run(streaks.to_numpy())
        
        # Basic stats
        features[f"{sensor}_mean"] = np.mean(x)
        features[f"{sensor}_std"] = np.std(x)
        features[f"{sensor}_min"] = np.min(x)
        features[f"{sensor}_max"] = np.max(x)
        features[f"{sensor}_iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
        features[f"{sensor}_cov"] = np.std(x) / (np.mean(x) + 1e-6)
        
        # Temporal dynamics
        features.update({
            f"{sensor}_{k}": v
            for k, v in derivative_features(x).items()
        })
        
        # Out-of-range
        min_val, max_val = ranges.get(sensor, (np.min(x), np.max(x)))
        features[f"{sensor}_oor_fraction"] = out_of_range_fraction(x, min_val, max_val)
        
        # Extreme jumps
        threshold = jump_thresholds.get(sensor, np.std(x) * 3)
        features[f"{sensor}_extreme_jump_fraction"] = extreme_jump_fraction(x, threshold)
        
        # Spectral entropy
        features[f"{sensor}_spectral_entropy"] = spectral_entropy(x)
        
        # Drift detection
        if baseline_stats and sensor in baseline_stats:
            base_mean, base_std = baseline_stats[sensor]
            features[f"{sensor}_mean_drift"] = abs(np.mean(x) - base_mean)
            features[f"{sensor}_std_drift"] = abs(np.std(x) - base_std)
    
    # -----------------------------------------------------
    # 2. Composite trust indicators
    # -----------------------------------------------------
    features["num_sensors_high_oor"] = sum(
        features[f"{s}_oor_fraction"] > 0.05 for s in df.columns
    )
    
    features["num_sensors_high_jump"] = sum(
        features[f"{s}_extreme_jump_fraction"] > 0.05 for s in df.columns
    )
    
    features["num_sensors_high_drift"] = sum(
        features.get(f"{s}_mean_drift", 0) > 0.1 for s in df.columns
    )
    
    return pd.DataFrame([features])

#build feature matrix
def build_feature_matrix(rw_list, ranges, jump_thresholds, baseline_stats):
    """
    rw_list: list of RW dataframes
    returns: feature matrix (DataFrame)
    """
    feature_rows = []
    for rw in rw_list:
        feats = extract_features(rw, ranges, jump_thresholds, baseline_stats)
        feature_rows.append(feats)
    return pd.concat(feature_rows, ignore_index=True)