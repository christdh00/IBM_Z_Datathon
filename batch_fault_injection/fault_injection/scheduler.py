"""Fault event scheduling for offline dataset benchmarks."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd

from .events import FaultEvent, REPLACEMENT_FAULTS


def validate_events(events: Iterable[FaultEvent], sensor_cols: Sequence[str], *, n_rows: int | None = None) -> tuple[FaultEvent, ...]:
    """Reject ambiguous overlaps and invalid events before processing."""
    events = tuple(events)
    ids = [e.event_id for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError("event_id values must be unique")
    for event in events:
        if event.sensor not in sensor_cols:
            raise ValueError(f"Unknown sensor {event.sensor!r}")
        if n_rows is not None and event.end > n_rows:
            raise ValueError(f"Event {event.event_id} exceeds dataset length")
    # Two replacement faults on the same sensor at once are ambiguous.
    repl = sorted((e for e in events if e.fault_type in REPLACEMENT_FAULTS),
                  key=lambda e: (e.sensor, e.start))
    latest_by_sensor: dict[str, FaultEvent] = {}
    for current in repl:
        previous = latest_by_sensor.get(current.sensor)
        if previous is not None and current.start < previous.end:
            raise ValueError(f"Overlapping replacement events: {previous.event_id}, {current.event_id}")
        latest_by_sensor[current.sensor] = current
    return events


def eligible_starts(
    df: pd.DataFrame, duration: int, *, reading_windows: Sequence[int] = (5, 10, 30, 60),
    failure_horizon_col: str = "PW_0.5h", timestamp_col: str = "timestamp",
    sampling_seconds: float = 60.0,
) -> np.ndarray:
    """Eligible indices with label 0 and contiguous timestamps in lookback/event.

    A 0 future-failure label does NOT certify a machine is currently healthy.
    Use with prerecorded validation or test data only.
    """
    duration = int(duration)
    if duration < 1 or duration > len(df):
        return np.empty(0, dtype=np.int64)
    if not reading_windows or any(int(w) < 1 for w in reading_windows):
        raise ValueError("reading_windows must contain positive integers")
    if failure_horizon_col not in df or timestamp_col not in df:
        raise KeyError("DataFrame needs timestamp and future-failure label for offline scheduling")
    if not df[failure_horizon_col].isin((0, 1)).all():
        raise ValueError("Future-failure labels must be binary 0/1")
    timestamps = pd.to_datetime(df[timestamp_col], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be ordered and unique")
    differences = timestamps.diff().dt.total_seconds().to_numpy()
    connected = np.r_[True, np.isclose(differences[1:], sampling_seconds)]
    bad = df[failure_horizon_col].to_numpy() != 0
    bad |= ~connected
    prefix = np.r_[0, np.cumsum(bad.astype(np.int64))]
    lookback = max(map(int, reading_windows)) - 1
    starts = np.arange(max(1, lookback), len(df) - duration + 1, dtype=np.int64)
    return starts[prefix[starts + duration] == prefix[starts - lookback]]


def schedule_random(
    df: pd.DataFrame, specs: Sequence[Mapping], *, sensor_cols: Sequence[str],
    seed: int = 42, reading_windows: Sequence[int] = (5, 10, 30, 60),
    allow_overlap: bool = False, min_gap: int = 0,
    failure_horizon_col: str = "PW_0.5h", timestamp_col: str = "timestamp",
    sampling_seconds: float = 60.0,
) -> list[FaultEvent]:
    """Schedule multiple injected events in one dataset.

    Specs: dict with sensor, fault_type, duration, severity_sd; optional
    event_id, seed, start. With allow_overlap=False, a common time timeline
    is protected for all sensors (plus min_gap readings on either side).
    For intentional cross-sensor or additive overlaps, allow_overlap=True or
    pass explicitly constructed FaultEvent objects to inject_batch.
    """
    if min_gap < 0:
        raise ValueError("min_gap must be >= 0")
    rng = np.random.default_rng(seed)
    events = []
    for index, spec in enumerate(specs, 1):
        duration = int(spec["duration"])
        event_id = str(spec.get("event_id", f"F{index:04d}"))
        event_seed = int(spec.get("seed", int(rng.integers(1, 2**31))))
        candidates = eligible_starts(df, duration, reading_windows=reading_windows,
            failure_horizon_col=failure_horizon_col, timestamp_col=timestamp_col,
            sampling_seconds=sampling_seconds)
        if spec["fault_type"] == "stuck_sensor":
            readings = df[spec["sensor"]].to_numpy(dtype=float)
            candidates = np.array([s for s in candidates if not np.all(
                np.isclose(readings[s:s+duration], readings[s-1], atol=1e-12, rtol=0.0))],dtype=np.int64)
        if not allow_overlap:
            candidates = np.array([s for s in candidates if all(
                s >= prev.end + min_gap or s + duration + min_gap <= prev.start
                for prev in events)], dtype=np.int64)
        if "start" in spec:
            start = int(spec["start"])
            if start not in candidates:
                raise ValueError(f"Requested start={start} invalid for {event_id}")
        else:
            if not len(candidates):
                raise ValueError(f"No suitable start for {event_id}; reduce duration/gap/count")
            start = int(rng.choice(candidates))
        events.append(FaultEvent(event_id=event_id, sensor=str(spec["sensor"]),
            fault_type=str(spec["fault_type"]), start=start, duration=duration,
            severity_sd=spec.get("severity_sd"), seed=event_seed))
        validate_events(events, sensor_cols, n_rows=len(df))
    return events
