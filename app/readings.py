from collections import deque
from dataclasses import dataclass


@dataclass
class Reading:
    index: int
    total: int
    timestamp: str
    cms_air_pressure: float
    oxygen_concentration: float
    nitrogen_pressure: float
    oxygen_over_threshold: float
    pw_0_5h: int
    window_rows: int
    window_required: int
    rows: list[dict] | None


class ReadingWindow:
    """Keep the readings the model needs and describe the current row."""

    def __init__(self, window_minutes: int) -> None:
        self.window_minutes = window_minutes
        self.rows: deque[dict] = deque(maxlen=window_minutes)
        self._last_index: int | None = None

    def observe(self, streamed: dict) -> Reading:
        index = streamed["index"]
        if self._last_index is not None and index <= self._last_index:
            self.rows.clear()
        self._last_index = index

        record = _record(streamed["record"])
        self.rows.append(record)
        ready = len(self.rows) >= self.window_minutes
        return Reading(
            index=index,
            total=streamed["total"],
            timestamp=record["timestamp"],
            cms_air_pressure=record["CMS air pressure"],
            oxygen_concentration=record["Concentrazione ossigeno base"],
            nitrogen_pressure=record["Nitrogen pressure"],
            oxygen_over_threshold=record["Oxygen over threshold"],
            pw_0_5h=record["PW_0.5h"],
            window_rows=len(self.rows),
            window_required=self.window_minutes,
            rows=list(self.rows) if ready else None,
        )


def _record(row: dict) -> dict:
    return {
        "timestamp": str(row["timestamp"]),
        "CMS air pressure": float(row["CMS air pressure"]),
        "Concentrazione ossigeno base": float(row["Concentrazione ossigeno base"]),
        "Nitrogen pressure": float(row["Nitrogen pressure"]),
        "Oxygen over threshold": float(row["Oxygen over threshold"]),
        "PW_0.5h": int(row["PW_0.5h"]),
    }
