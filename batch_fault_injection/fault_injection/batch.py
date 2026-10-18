"""Offline DataFrame injection only. No streaming state or live transport code."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json

import numpy as np
import pandas as pd

from .events import FaultEvent, ADDITIVE_FAULTS, REPLACEMENT_FAULTS
from .faults import prepare_fault
from .scheduler import validate_events

ANNOTATION_COLUMNS = (
    "injection_interval", "synthetic_anomaly", "synthetic_modified",
    "special_fault_active", "benign_noise_active", "intermittent_active",
    "overlap_count", "synthetic_event_id", "anomaly_id",
    "synthetic_event_ids", "anomaly_ids",
)


def inject_batch(
    df: pd.DataFrame,
    events: Sequence[FaultEvent],
    *,
    reference_stats: Mapping[str, Mapping[str, float]],
    sensor_cols: Sequence[str],
    failure_horizon_col: str = "PW_0.5h",
    timestamp_col: str = "timestamp",
    require_no_failure_horizon: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply multiple faults with deterministic composition and retain ground truth.

    Priority on one sensor: additive offsets -> one replacement -> dropout.
    Two overlapping replacements on the same sensor are rejected.
    Existing `PW_0.5h` is preserved. The original frame is never modified.

    synthetic_modified = actual sensor-row numerical difference (incl. benign).
    synthetic_anomaly = presence of any *intended* true fault activity.
    special_fault_active = benign Gaussian OR active intermittent fault.
    Overlap count = number of scheduled events, including inactive phases.
    """
    sensor_cols = tuple(sensor_cols)
    if any(c in df.columns for c in ANNOTATION_COLUMNS):
        raise ValueError("Input is already annotated; provide the original unmodified source")
    if not sensor_cols or len(set(sensor_cols)) != len(sensor_cols):
        raise ValueError("sensor_cols must be a nonempty unique list")
    events = validate_events(events, sensor_cols, n_rows=len(df))
    if require_no_failure_horizon:
        if failure_horizon_col not in df or not df[failure_horizon_col].isin((0, 1)).all():
            raise ValueError(f"Need binary {failure_horizon_col} for offline safeguards")
    original = df.reset_index(drop=True)
    n = len(original)
    for sensor in sensor_cols:
        if sensor not in original:
            raise KeyError(f"Missing sensor {sensor}")
        values = pd.to_numeric(original[sensor], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Source {sensor} includes missing or nonfinite values")
    if require_no_failure_horizon:
        pre_failure = original[failure_horizon_col].to_numpy() != 0
        for event in events:
            if pre_failure[event.start:event.end].any():
                raise ValueError(f"{event.event_id} intersects labeled impending-failure rows")

    result = original.copy(deep=True)
    by_sensor: dict[str, list[FaultEvent]] = defaultdict(list)
    per_row_events: list[list[str]] = [[] for _ in range(n)]
    per_row_anomalies: list[list[str]] = [[] for _ in range(n)]
    special = np.zeros(n, dtype=bool)
    benign = np.zeros(n, dtype=bool)
    intermittent = np.zeros(n, dtype=bool)
    for event in events:
        by_sensor[event.sensor].append(event)
        for i in range(event.start, event.end):
            per_row_events[i].append(event.event_id)
            active = event.fault_type != "gaussian_noise" and (
                event.fault_type != "intermittent_fault" or (i-event.start) % 2 == 0
            )
            if active:
                per_row_anomalies[i].append(event.event_id)
            if event.fault_type == "gaussian_noise":
                benign[i] = True
                special[i] = True
            if event.fault_type == "intermittent_fault" and active:
                intermittent[i] = True
                special[i] = True

    altered = np.zeros(n, dtype=bool)
    for sensor, sensor_events in by_sensor.items():
        raw = original[sensor].to_numpy(dtype=float)
        output = raw.copy()
        plans = {e.event_id: prepare_fault(e, reference_stats) for e in sensor_events}
        for event in sorted(sensor_events, key=lambda e: e.event_id):
            if event.fault_type in ADDITIVE_FAULTS:
                output[event.start:event.end] += plans[event.event_id].offsets
        for event in sorted(sensor_events, key=lambda e: e.event_id):
            if event.fault_type == "random_corruption":
                output[event.start:event.end] = plans[event.event_id].offsets
            elif event.fault_type == "stuck_sensor":
                output[event.start:event.end] = raw[event.start-1]
        for event in sensor_events:
            if event.fault_type == "dropout":
                output[event.start:event.end] = np.nan
        altered |= ~np.isclose(output, raw, rtol=0, atol=1e-12, equal_nan=True)
        result[sensor] = output

    result["injection_interval"] = np.array([bool(x) for x in per_row_events], dtype="int8")
    result["synthetic_anomaly"] = np.array([bool(x) for x in per_row_anomalies], dtype="int8")
    result["synthetic_modified"] = altered.astype("int8")
    result["special_fault_active"] = special.astype("int8")
    result["benign_noise_active"] = benign.astype("int8")
    result["intermittent_active"] = intermittent.astype("int8")
    result["overlap_count"] = np.array([len(x) for x in per_row_events], dtype="int16")
    result["synthetic_event_id"] = ["|".join(sorted(x)) for x in per_row_events]
    result["anomaly_id"] = ["|".join(sorted(x)) for x in per_row_anomalies]
    result["synthetic_event_ids"] = [json.dumps(sorted(x)) for x in per_row_events]
    result["anomaly_ids"] = [json.dumps(sorted(x)) for x in per_row_anomalies]

    records = []
    for event in events:
        start, end = event.start, event.end
        source_values = original[event.sensor].iloc[start:end].to_numpy(dtype=float)
        final_values = result[event.sensor].iloc[start:end].to_numpy(dtype=float)
        records.append({
            **event.to_dict(),
            "start_timestamp": str(original[timestamp_col].iloc[start]) if timestamp_col in original else "",
            "end_timestamp": str(original[timestamp_col].iloc[end-1]) if timestamp_col in original else "",
            "intended_anomalous_readings": 0 if event.fault_type == "gaussian_noise"
                else (end-start+1)//2 if event.fault_type == "intermittent_fault" else end-start,
            "final_sensor_rows_different": int((~np.isclose(
                final_values, source_values, rtol=0, atol=1e-12, equal_nan=True)).sum()),
            "is_negative_control": int(event.fault_type == "gaussian_noise"),
        })
    return result, pd.DataFrame.from_records(records)


class FaultEngine:
    """Small batch-only wrapper compatible with the previous FaultEngine usage.

    ``process_one`` is deliberately not provided: this project no longer
    contains or maintains streaming-specific code.
    """
    def __init__(self, reference_stats, events, *, sensor_cols):
        self.reference_stats = reference_stats
        self.events = tuple(events)
        self.sensor_cols = tuple(sensor_cols)
        self._last_manifest = None

    def process_batch(self, df, *, failure_horizon_col="PW_0.5h",
                      require_no_failure_horizon=True):
        result, self._last_manifest = inject_batch(
            df, self.events, reference_stats=self.reference_stats,
            sensor_cols=self.sensor_cols, failure_horizon_col=failure_horizon_col,
            require_no_failure_horizon=require_no_failure_horizon,
        )
        return result

    def event_manifest(self, original=None, modified=None):
        if self._last_manifest is None:
            raise RuntimeError("Call process_batch before event_manifest")
        return self._last_manifest.copy(deep=True)
