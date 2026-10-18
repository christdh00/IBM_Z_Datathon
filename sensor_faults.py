"""Reusable synthetic sensor-fault injection utilities.

Adapted from synthetic_sensor_fault_benchmarks.ipynb.

Typical workflow:
    stats = compute_reference_stats(train_df, sensors, failure_horizon_col="PW_0.5h")
    start = choose_start(test_df, sensors[0], "spike", 1, 43, reading_windows=(5, 10, 30, 60))
    modified, event = inject_fault(test_df, sensors[0], "spike", start, 1,
                                   2.0, 43, "E000001", reference_stats=stats,
                                   sensor_cols=sensors)

The data frame passed to inject_fault is never mutated. For evaluation,
calculate reference statistics from training data (not held-out test data).
"""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
import pandas as pd

FAULT_TYPES = (
    "gaussian_noise", "spike", "short_burst", "random_corruption", "dropout",
    "bias", "drift", "stuck_sensor", "increased_noise", "intermittent_fault",
)


def compute_reference_stats(
    training_df: pd.DataFrame,
    sensor_cols: Sequence[str],
    *,
    failure_horizon_col: str = "PW_0.5h",
) -> dict[str, dict[str, float]]:
    """Compute sensor mean/SD/range from label=0 TRAINING readings only.

    A zero label means *not within the labeled pre-failure horizon*, not
    necessarily verified healthy operation.
    """
    if failure_horizon_col not in training_df.columns:
        raise KeyError(f"Missing failure-horizon column: {failure_horizon_col}")
    if not training_df[failure_horizon_col].isin([0, 1]).all():
        raise ValueError("Future-failure labels must be binary 0/1")
    reference_df = training_df.loc[training_df[failure_horizon_col].eq(0)]
    if len(reference_df) < 3:
        raise ValueError("At least three training reference readings are required")
    result = {}
    for sensor in sensor_cols:
        if sensor not in training_df.columns:
            raise KeyError(f"Missing sensor: {sensor}")
        values = pd.to_numeric(reference_df[sensor], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Reference sensor has missing/nonfinite values: {sensor}")
        sd = float(np.std(values, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Sensor has zero/invalid reference SD: {sensor}")
        result[sensor] = {
            "mean": float(np.mean(values)), "std": sd,
            "min": float(np.min(values)), "max": float(np.max(values)),
        }
    return result


def eligible_starts(
    df: pd.DataFrame,
    duration: int,
    *,
    reading_windows: Sequence[int] = (5, 10, 30, 60),
    failure_horizon_col: str = "PW_0.5h",
    timestamp_col: str = "timestamp",
    sampling_seconds: float = 60.0,
) -> np.ndarray:
    """Find starts where both the event and preceding RW history have label 0.

    Requires consecutive samples (timestamps spaced by sampling_seconds).
    Uses zero-based row *positions* rather than original DataFrame index labels.
    """
    duration = int(duration)
    if duration < 1 or duration > len(df):
        return np.empty(0, dtype=np.int64)
    if not reading_windows or any(int(w) < 1 for w in reading_windows):
        raise ValueError("Reading windows must contain positive integers")
    if failure_horizon_col not in df or timestamp_col not in df:
        raise KeyError("Missing future-failure label or timestamp column")
    if not df[failure_horizon_col].isin([0, 1]).all():
        raise ValueError("Future-failure labels must be binary 0/1")
    ts = pd.to_datetime(df[timestamp_col], errors="raise")
    steps = ts.diff().dt.total_seconds().to_numpy()
    regular_step = np.r_[True, np.isclose(steps[1:], sampling_seconds)]
    bad = df[failure_horizon_col].to_numpy() != 0
    bad = bad | ~regular_step
    prefix = np.r_[0, np.cumsum(bad.astype(np.int64))]
    lookback = max(map(int, reading_windows)) - 1
    starts = np.arange(max(1, lookback), len(df) - duration + 1, dtype=np.int64)
    # Keep exactly the lookback and fault interval clear of flagged points.
    return starts[prefix[starts + duration] == prefix[starts - lookback]]


def choose_start(
    df: pd.DataFrame,
    sensor: str,
    fault_type: str,
    duration: int,
    seed: int,
    *,
    reading_windows: Sequence[int] = (5, 10, 30, 60),
    failure_horizon_col: str = "PW_0.5h",
    timestamp_col: str = "timestamp",
    sampling_seconds: float = 60.0,
    random_seed: int = 42,
) -> int:
    """Reproducibly choose an eligible fault start, including stuck-sensor guard."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unknown fault type: {fault_type}")
    starts = eligible_starts(
        df, duration, reading_windows=reading_windows,
        failure_horizon_col=failure_horizon_col,
        timestamp_col=timestamp_col, sampling_seconds=sampling_seconds,
    )
    if fault_type == "stuck_sensor":
        arr = df[sensor].to_numpy(dtype=float)
        starts = np.asarray([
            s for s in starts if not np.all(np.isclose(
                arr[s:s + duration], arr[s - 1], atol=1e-12, rtol=0))
        ], dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"No eligible injection location for {sensor}/{fault_type}, duration={duration}")
    rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), int(seed), 991]))
    return int(rng.choice(starts))


def inject_fault(
    clean_df: pd.DataFrame,
    sensor: str,
    fault_type: str,
    start_index: int,
    duration: int,
    severity_sd: float | None,
    seed: int,
    event_id: str,
    *,
    reference_stats: Mapping[str, Mapping[str, float]],
    sensor_cols: Sequence[str],
    failure_horizon_col: str = "PW_0.5h",
    timestamp_col: str = "timestamp",
    benchmark_split: str = "unspecified",
    reference_source: str = "training readings with future-failure label 0",
) -> tuple[pd.DataFrame, dict]:
    """Inject one event with the same ten fault definitions as the source notebook.

    Caller should choose a safe start using `choose_start`. Positions refer to
    rows, regardless of their pandas index labels. Returns (new_df, metadata).
    """
    if sensor not in sensor_cols or fault_type not in FAULT_TYPES:
        raise ValueError("Unexpected sensor or fault type")
    if sensor not in reference_stats:
        raise KeyError(f"Missing reference statistics for sensor: {sensor}")
    if failure_horizon_col not in clean_df:
        raise KeyError(f"Missing future-failure label: {failure_horizon_col}")
    start_index, duration = int(start_index), int(duration)
    end_exclusive = start_index + duration
    if duration < 1 or start_index < 1 or end_exclusive > len(clean_df):
        raise ValueError("Invalid fault interval")
    if not clean_df[failure_horizon_col].iloc[start_index:end_exclusive].eq(0).all():
        raise ValueError("Cannot inject during a positive pre-failure interval")
    if fault_type in ("dropout", "stuck_sensor"):
        severity_sd = None
    elif severity_sd is None or not np.isfinite(float(severity_sd)) or float(severity_sd) <= 0:
        raise ValueError("Positive severity required")
    rng = np.random.default_rng(int(seed))
    # Normalize index to positions so .loc follows .iloc throughout.
    out = clean_df.copy(deep=True).reset_index(drop=True)
    values = out[sensor].to_numpy(dtype=float).copy()
    baseline = values[start_index:end_exclusive].copy()
    sd = float(reference_stats[sensor]["std"])
    signed_direction = int(rng.choice([-1, 1]))
    amplitude = float(severity_sd) * sd if severity_sd is not None else None
    positive_phase = np.ones(duration, dtype=bool)

    if fault_type == "gaussian_noise":
        values[start_index:end_exclusive] += rng.normal(0, amplitude, duration)
        positive_phase[:] = False  # benign negative-control jitter
    elif fault_type == "spike":
        if duration != 1:
            raise ValueError("Spike must have duration 1")
        values[start_index] += signed_direction * amplitude
    elif fault_type == "short_burst":
        if duration < 3:
            raise ValueError("Short burst needs at least 3 readings")
        values[start_index:end_exclusive] += signed_direction * amplitude
    elif fault_type == "random_corruption":
        offsets = rng.uniform(1.0, 2.0, duration) * amplitude
        signs = rng.choice([-1.0, 1.0], size=duration)
        values[start_index:end_exclusive] = reference_stats[sensor]["mean"] + signs * offsets
    elif fault_type == "dropout":
        values[start_index:end_exclusive] = np.nan
    elif fault_type == "bias":
        values[start_index:end_exclusive] += signed_direction * amplitude
    elif fault_type == "drift":
        ramp = np.linspace(0.0, 1.0, duration + 1)[1:]
        values[start_index:end_exclusive] += signed_direction * amplitude * ramp
    elif fault_type == "stuck_sensor":
        values[start_index:end_exclusive] = values[start_index - 1]
    elif fault_type == "increased_noise":
        values[start_index:end_exclusive] += rng.normal(0, amplitude, duration)
    elif fault_type == "intermittent_fault":
        positive_phase = np.arange(duration) % 2 == 0
        slc = values[start_index:end_exclusive].copy()
        slc[positive_phase] += signed_direction * amplitude
        values[start_index:end_exclusive] = slc

    changed_within = ~np.isclose(values[start_index:end_exclusive], baseline,
                                 atol=1e-12, rtol=0.0, equal_nan=True)
    if fault_type != "gaussian_noise" and not changed_within.any():
        raise ValueError("Invisible fault (no observable sensor change)")

    out[sensor] = values
    out["injection_interval"] = np.zeros(len(out), dtype="int8")
    out.loc[start_index:end_exclusive - 1, "injection_interval"] = 1
    out["synthetic_anomaly"] = np.zeros(len(out), dtype="int8")
    out.loc[start_index:end_exclusive - 1, "synthetic_anomaly"] = positive_phase.astype("int8")
    out["synthetic_event_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[start_index:end_exclusive - 1, "synthetic_event_id"] = event_id
    out["anomaly_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    positive_indices = start_index + np.flatnonzero(positive_phase)
    out.loc[positive_indices, "anomaly_id"] = event_id

    metadata = {
        "experiment_id": event_id, "sensor": sensor, "fault_type": fault_type,
        "severity_sd": severity_sd, "duration": duration,
        "start_index": start_index, "end_index": end_exclusive - 1,
        "start_timestamp": str(out.at[start_index, timestamp_col]),
        "end_timestamp": str(out.at[end_exclusive - 1, timestamp_col]),
        "seed": int(seed), "direction": signed_direction if amplitude is not None else None,
        "n_labeled_readings": int(positive_phase.sum()),
        "n_changed_readings": int(changed_within.sum()),
        "is_negative_control": int(fault_type == "gaussian_noise"),
        "reference_source": reference_source, "benchmark_split": benchmark_split,
        "baseline_sd": sd, "benchmark_csv": "",
    }
    return out, metadata


def align_reading_window_labels(
    benchmark_df: pd.DataFrame, window: int, *,
    failure_horizon_col: str = "PW_0.5h", timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Align point and any-in-window anomaly labels at each causal window end."""
    window = int(window)
    if not 1 <= window <= len(benchmark_df):
        raise ValueError("Bad reading-window size")
    label = benchmark_df["synthetic_anomaly"].to_numpy(dtype=np.int8)
    state = benchmark_df[failure_horizon_col].to_numpy(dtype=np.int8)
    end = np.arange(window - 1, len(benchmark_df), dtype=int)
    positives = np.r_[0, np.cumsum(label)]
    positive_horizon_prefix = np.r_[0, np.cumsum(state == 1)]
    labels_in_window = positives[end + 1] - positives[end + 1 - window]
    positive_horizon_count = positive_horizon_prefix[end + 1] - positive_horizon_prefix[end + 1 - window]
    return pd.DataFrame({
        "window": window,
        "timestamp_end": benchmark_df[timestamp_col].iloc[end].to_numpy(),
        "reading_index_end": end,
        "synthetic_anomaly_at_end": label[end],
        "synthetic_anomaly_any_in_window": (labels_in_window > 0).astype("int8"),
        "no_positive_failure_horizon_labels_in_window": (positive_horizon_count == 0).astype("int8"),
    })


def regenerate_from_manifest(
    df: pd.DataFrame, row: Mapping, *, reference_stats: Mapping[str, Mapping[str, float]],
    sensor_cols: Sequence[str], failure_horizon_col: str = "PW_0.5h",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Recreate a benchmark using an experiment manifest record."""
    severity = row["severity_sd"]
    severity = None if pd.isna(severity) else float(severity)
    modified, _ = inject_fault(
        df, row["sensor"], row["fault_type"], int(row["start_index"]),
        int(row["duration"]), severity, int(row["seed"]), str(row["experiment_id"]),
        reference_stats=reference_stats, sensor_cols=sensor_cols,
        failure_horizon_col=failure_horizon_col, timestamp_col=timestamp_col,
        benchmark_split=str(row.get("benchmark_split", "unspecified")),
    )
    return modified
