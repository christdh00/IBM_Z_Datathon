"""Generate offline mixed-fault datasets with controlled modified-row percentages."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd

from .events import FaultEvent
from .scheduler import eligible_starts
from .batch import inject_batch

DEFAULT_FAULT_WEIGHTS = {
    "spike": 1, "short_burst": 1, "random_corruption": 1, "dropout": 1,
    "bias": 1, "drift": 1, "stuck_sensor": 1, "increased_noise": 1,
}


def compute_reference_stats(training_df: pd.DataFrame, sensor_cols: Sequence[str], *,
                            failure_horizon_col: str = "PW_0.5h") -> dict:
    """Fit intensity statistics ONLY on train rows with pre-failure label zero."""
    if failure_horizon_col not in training_df:
        raise KeyError(f"Missing {failure_horizon_col}")
    if not training_df[failure_horizon_col].isin([0, 1]).all():
        raise ValueError("Future-failure labels must be 0 or 1")
    baseline = training_df.loc[training_df[failure_horizon_col].eq(0)]
    if len(baseline) < 3:
        raise ValueError("Insufficient baseline readings")
    stats = {}
    for name in sensor_cols:
        x = pd.to_numeric(baseline[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"Training sensor {name} contains nonfinite readings")
        sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Sensor {name} has zero or invalid SD")
        stats[name] = {"mean": float(x.mean()), "std": sd,
                       "min": float(x.min()), "max": float(x.max())}
    return stats


def _schedule_primary(
    df: pd.DataFrame, *, target_count: int, special_share: float,
    gaussian_share: float, seed: int, sensor_cols: Sequence[str],
    reading_windows: Sequence[int], durations: Sequence[int],
    severities_sd: Sequence[float], min_gap: int,
    failure_horizon_col: str, fault_weights: Mapping[str, float],
) -> tuple[list[FaultEvent], int]:
    rng = np.random.default_rng(seed)
    n = len(df)
    if target_count > n:
        raise ValueError("Target count exceeds number of readings")
    if not (0 <= special_share <= 1 and 0 <= gaussian_share <= 1):
        raise ValueError("Special and Gaussian shares must be within [0, 1]")
    if min_gap < 0:
        raise ValueError("min_gap must be nonnegative")
    if not durations or any(int(d) < 1 for d in durations):
        raise ValueError("Durations must be positive")
    if not severities_sd or any(float(x) <= 0 for x in severities_sd):
        raise ValueError("Severities must be positive")
    if not fault_weights or any(kind not in DEFAULT_FAULT_WEIGHTS or weight < 0
                                for kind, weight in fault_weights.items()) or not sum(fault_weights.values()):
        raise ValueError("Invalid ordinary fault weights")
    events: list[FaultEvent] = []
    reserved = np.zeros(n, dtype=bool)  # full intervals, even intermittent off-phases
    raw = {s: df[s].to_numpy(dtype=float) for s in sensor_cols}
    cache: dict[int, np.ndarray] = {}

    def starts_for(duration: int, sensor: str, kind: str) -> np.ndarray:
        if duration not in cache:
            cache[duration] = eligible_starts(
                df, duration, reading_windows=reading_windows,
                failure_horizon_col=failure_horizon_col)
        starts = cache[duration]
        occ = np.r_[0, np.cumsum(reserved.astype(np.int64))]
        lo = np.maximum(0, starts-min_gap)
        hi = np.minimum(n, starts+duration+min_gap)
        starts = starts[occ[hi] == occ[lo]]
        if kind == "stuck_sensor":
            # Avoid scheduling a nominally stuck interval with unchanged samples.
            x = raw[sensor]
            starts = np.asarray([
                start for start in starts if np.all(~np.isclose(
                    x[start:start+duration], x[start-1], atol=1e-12, rtol=0))
            ], dtype=np.int64)
        return starts

    def add(kind: str, active_count: int, *, trials: int = 20) -> bool:
        # Intermittent modifies only even offsets; use twice the duration.
        duration = 2*active_count if kind == "intermittent_fault" else active_count
        if kind == "short_burst" and duration < 3:
            return False
        if kind == "spike" and duration != 1:
            return False
        for _ in range(trials):
            sensor = str(rng.choice(sensor_cols))
            candidates = starts_for(duration, sensor, kind)
            if len(candidates) == 0:
                continue
            start = int(rng.choice(candidates))
            severity = None if kind in ("dropout", "stuck_sensor") else float(rng.choice(severities_sd))
            if kind == "gaussian_noise":
                # Benign jitter: small SD, not a severe fault amplitude.
                severity = float(rng.choice((0.02, 0.05, 0.1)))
            event = FaultEvent(
                event_id=f"F{len(events)+1:05d}", sensor=sensor,
                fault_type=kind, start=start, duration=duration,
                severity_sd=severity, seed=int(rng.integers(0, 2**31-1)),
            )
            events.append(event)
            reserved[start:start+duration] = True
            return True
        return False

    special_count = int(round(target_count * special_share))
    gaussian_count = int(round(special_count * gaussian_share))
    intermittent_count = special_count - gaussian_count

    # Reserve 5–10% of *modified rows* for small Gaussian jitter and
    # active intermittent-fault phases; intermittent off-phases stay unmodified.
    for kind, count in (("gaussian_noise", gaussian_count), ("intermittent_fault", intermittent_count)):
        remaining = count
        while remaining:
            max_chunk = min(30 if kind == "gaussian_noise" else 15, remaining)
            chunk = int(rng.integers(1, max_chunk+1))
            if not add(kind, chunk):
                if not add(kind, 1):
                    raise RuntimeError(f"Unable to place special fault {kind}; remaining={remaining}")
                chunk = 1
            remaining -= chunk

    remaining = target_count - special_count
    kinds = list(fault_weights)
    weights = np.asarray(list(fault_weights.values()), dtype=float)
    weights /= weights.sum()
    while remaining:
        placed = False
        for _ in range(50):
            kind = "spike" if remaining < 3 else str(rng.choice(kinds, p=weights))
            if kind == "spike":
                chunk = 1
            else:
                valid = [int(d) for d in durations if d <= remaining and
                         (kind != "short_burst" or d >= 3)]
                if not valid:
                    continue
                chunk = int(rng.choice(valid))
            if add(kind, chunk, trials=3):
                remaining -= chunk
                placed = True
                break
        if not placed:
            if add("spike", 1, trials=40):
                remaining -= 1
            else:
                raise RuntimeError(
                    f"Cannot place {remaining} more changed rows in eligible intervals. "
                    "Reduce contamination, min_gap, or history requirements."
                )
    return events, special_count


def _add_overlaps(
    primary: list[FaultEvent], *, sensor_cols: Sequence[str], seed: int,
    overlap_event_fraction: float, same_sensor_overlap_probability: float,
    severity_values: Sequence[float],
) -> list[FaultEvent]:
    if not (0 <= overlap_event_fraction <= 1 and 0 <= same_sensor_overlap_probability <= 1):
        raise ValueError("Overlap controls must be in [0, 1]")
    rng = np.random.default_rng(seed)
    events = list(primary)
    ordinary = [e for e in primary if e.fault_type not in ("gaussian_noise", "intermittent_fault")]
    n_extra = int(round(len(ordinary) * overlap_event_fraction))
    if overlap_event_fraction > 0 and ordinary:
        n_extra = max(1, n_extra)
    indices = rng.choice(len(ordinary), size=n_extra, replace=False) if n_extra else []
    for chosen_index in indices:
        parent = ordinary[int(chosen_index)]
        # Secondary events are wholly nested within ordinary events, so they
        # increase multi-fault complexity without increasing modified-row budget.
        same_sensor = rng.random() < same_sensor_overlap_probability
        if same_sensor and parent.fault_type in {"dropout", "stuck_sensor", "random_corruption"}:
            same_sensor = False
        possibilities = [parent.sensor] if same_sensor else [s for s in sensor_cols if s != parent.sensor]
        if not possibilities:
            continue
        sensor = str(rng.choice(possibilities))
        kind = str(rng.choice(("bias", "increased_noise", "drift", "spike", "dropout")))
        if same_sensor and kind == "dropout":
            kind = "increased_noise"  # prevent masking all prior additive effects
        duration = 1 if kind == "spike" else int(rng.integers(1, min(parent.duration, 12)+1))
        if kind == "spike":
            duration = 1
        start = int(rng.integers(parent.start, parent.end-duration+1))
        events.append(FaultEvent(
            event_id=f"F{len(events)+1:05d}", sensor=sensor,
            fault_type=kind, start=start, duration=duration,
            severity_sd=None if kind == "dropout" else float(rng.choice(severity_values)),
            seed=int(rng.integers(0, 2**31-1)),
        ))
    return events


def generate_contamination_dataset(
    source_df: pd.DataFrame,
    *,
    reference_stats: Mapping[str, Mapping[str, float]],
    sensor_cols: Sequence[str],
    target_percent: float,
    special_share: float = 0.075,
    gaussian_share_of_special: float = 0.5,
    overlap_event_fraction: float = 0.20,
    same_sensor_overlap_probability: float = 0.20,
    reading_windows: Sequence[int] = (5, 10, 30, 60),
    durations: Sequence[int] = (3, 5, 10, 30, 60),
    severities_sd: Sequence[float] = (0.5, 1, 2, 3),
    fault_weights: Mapping[str, float] | None = None,
    min_gap: int = 0,
    failure_horizon_col: str = "PW_0.5h",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (modified_frame, event_manifest, summary) for a target modified-row %.

    The target is DISTINCT ROWS WITH ANY NUMERICAL SENSOR CHANGE, not number
    of events or synthetic_anomaly=1 labels. Gaussian is a benign negative
    control, and intermittent has alternating active/nonactive readings.
    Special share counts changed rows involving those types (non-overlapping
    with ordinary primary events by default). Extra overlaps are nested only
    inside ordinary primary events. A failed exact-target check raises.
    """
    if not 0 <= target_percent <= 100:
        raise ValueError("target_percent must be between 0 and 100")
    if not source_df.index.is_unique:
        raise ValueError("Source index must be unique")
    target_count = int(round(len(source_df)*target_percent/100))
    ordinary = DEFAULT_FAULT_WEIGHTS if fault_weights is None else fault_weights
    for attempt in range(4):
        primary, requested_special = _schedule_primary(
            source_df, target_count=target_count, special_share=special_share,
            gaussian_share=gaussian_share_of_special, seed=seed+attempt*1009,
            sensor_cols=sensor_cols, reading_windows=reading_windows,
            durations=durations, severities_sd=severities_sd,
            min_gap=min_gap, failure_horizon_col=failure_horizon_col,
            fault_weights=ordinary,
        )
        events = _add_overlaps(
            primary, sensor_cols=sensor_cols, seed=seed+attempt*1009+789,
            overlap_event_fraction=overlap_event_fraction,
            same_sensor_overlap_probability=same_sensor_overlap_probability,
            severity_values=severities_sd,
        )
        data, manifest = inject_batch(
            source_df, events, reference_stats=reference_stats,
            sensor_cols=sensor_cols, failure_horizon_col=failure_horizon_col,
        )
        modified = data["synthetic_modified"].to_numpy().astype(bool)
        achieved = int(modified.sum())
        special_count = int((modified & data["special_fault_active"].to_numpy().astype(bool)).sum())
        if achieved == target_count and special_count == requested_special:
            break
    else:
        raise RuntimeError(
            f"Cannot achieve exact changed-row budget after retries: "
            f"requested {target_count}, last produced {achieved}."
        )
    true_anomaly_count = int(data["synthetic_anomaly"].sum())
    benign_count = int((modified & data["benign_noise_active"].to_numpy().astype(bool)).sum())
    intermittent_count = int((modified & data["intermittent_active"].to_numpy().astype(bool)).sum())
    summary = {
        "target_modified_percent": float(target_percent),
        "actual_modified_percent": 100*achieved/len(data),
        "modified_rows": achieved,
        "target_modified_rows": target_count,
        "anomalous_rows": true_anomaly_count,
        "actual_anomaly_percent": 100*true_anomaly_count/len(data),
        "special_modified_rows": special_count,
        "actual_special_share_percent": 100*special_count/achieved if achieved else 0.0,
        "benign_modified_rows": benign_count,
        "intermittent_modified_rows": intermittent_count,
        "overlapping_rows": int((data["overlap_count"] > 1).sum()),
        "event_count": len(events), "seed": seed, "total_rows": len(data),
    }
    return data, manifest, summary
