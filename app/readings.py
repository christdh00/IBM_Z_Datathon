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
    rows: list[dict]


class ReadingWindow:
    """Keep recent readings and describe the current row.

    ``window_minutes`` is the count reported as required. ``buffer_minutes``
    is how many rows are retained, so a longer model can read the same stream.
    """

    def __init__(self, window_minutes: int, buffer_minutes: int | None = None) -> None:
        history = window_minutes if buffer_minutes is None else buffer_minutes
        if history < window_minutes:
            raise ValueError("buffer_minutes must be at least window_minutes")
        self.window_minutes = window_minutes
        self.rows: deque[dict] = deque(maxlen=history)
        self._last_index: int | None = None

    def observe(self, streamed: dict) -> Reading:
        index = streamed["index"]
        if self._last_index is not None and index <= self._last_index:
            self.rows.clear()
        self._last_index = index

        record = _record(streamed["record"])
        self.rows.append(record)
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
            rows=list(self.rows),
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
