"""Fault event definitions shared by offline and online injection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

FAULT_TYPES = (
    "gaussian_noise", "spike", "short_burst", "random_corruption", "dropout",
    "bias", "drift", "stuck_sensor", "increased_noise", "intermittent_fault",
)

NEGATIVE_CONTROLS = {"gaussian_noise"}
ADDITIVE_FAULTS = {
    "gaussian_noise", "spike", "short_burst", "bias", "drift",
    "increased_noise", "intermittent_fault",
}
REPLACEMENT_FAULTS = {"random_corruption", "stuck_sensor"}


@dataclass(frozen=True)
class FaultEvent:
    """One scheduled sensor fault; indices are zero-based *reading positions*.

    ``start`` is inclusive; ``end`` is exclusive. ``seed`` is local to this
    event: adding another event does not alter its generated noise.
    """
    event_id: str
    sensor: str
    fault_type: str
    start: int
    duration: int
    severity_sd: float | None
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a nonempty string")
        if not isinstance(self.sensor, str) or not self.sensor.strip():
            raise ValueError("sensor must be a nonempty string")
        if self.fault_type not in FAULT_TYPES:
            raise ValueError(f"Unsupported fault_type: {self.fault_type}")
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start must be a nonnegative integer reading index")
        if isinstance(self.duration, bool) or not isinstance(self.duration, int) or self.duration < 1:
            raise ValueError("duration must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.fault_type == "spike" and self.duration != 1:
            raise ValueError("spike requires duration=1")
        if self.fault_type == "short_burst" and self.duration < 3:
            raise ValueError("short_burst requires duration>=3")
        if self.fault_type == "intermittent_fault" and self.duration < 2:
            raise ValueError("intermittent_fault requires duration>=2")
        if self.fault_type == "stuck_sensor" and self.start == 0:
            raise ValueError("stuck_sensor requires at least one prior reading")
        if self.fault_type in {"stuck_sensor", "dropout"}:
            if self.severity_sd is not None:
                raise ValueError(f"{self.fault_type} uses severity_sd=None")
        elif (self.severity_sd is None or isinstance(self.severity_sd, bool)
              or not math.isfinite(float(self.severity_sd)) or float(self.severity_sd) <= 0):
            raise ValueError(f"{self.fault_type} requires positive finite severity_sd")

    @property
    def end(self) -> int:
        return self.start + self.duration

    def active_at(self, index: int) -> bool:
        return self.start <= index < self.end

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "end_exclusive": self.end}
