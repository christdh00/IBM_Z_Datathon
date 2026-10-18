"""Pure, deterministic per-event fault transformations.

Reuses the ten fault rules from the user's single-event benchmark notebook.
Each event precomputes its random offsets once so offline batch generation
is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from .events import FaultEvent, NEGATIVE_CONTROLS


@dataclass(frozen=True)
class FaultPlan:
    event: FaultEvent
    offsets: np.ndarray
    direction: int

    def is_positive(self, position: int) -> bool:
        """Ground-truth fault activity, not whether output differs numerically."""
        if self.event.fault_type in NEGATIVE_CONTROLS:
            return False
        if self.event.fault_type == "intermittent_fault":
            return position % 2 == 0
        return True


def prepare_fault(event: FaultEvent, reference_stats: Mapping[str, Mapping[str, float]]) -> FaultPlan:
    """Precalculate stochastic draws, matching the original one-event rules."""
    if event.sensor not in reference_stats:
        raise KeyError(f"Missing reference statistics for {event.sensor!r}")
    stats = reference_stats[event.sensor]
    sd = float(stats["std"])
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Invalid reference standard deviation for {event.sensor}")
    rng = np.random.default_rng(event.seed)
    direction = int(rng.choice([-1, 1]))  # same first draw as original notebook
    amplitude = None if event.severity_sd is None else float(event.severity_sd) * sd
    length = event.duration
    kind = event.fault_type
    offsets = np.zeros(length, dtype=float)
    if kind in {"gaussian_noise", "increased_noise"}:
        offsets = rng.normal(0, amplitude, length)
    elif kind == "random_corruption":
        distances = rng.uniform(1.0, 2.0, length) * amplitude
        signs = rng.choice([-1.0, 1.0], size=length)
        offsets = float(stats["mean"]) + signs * distances  # absolute replacement values
    elif kind == "drift":
        offsets = direction * amplitude * np.linspace(0.0, 1.0, length + 1)[1:]
    elif kind == "intermittent_fault":
        offsets[::2] = direction * amplitude
    elif kind in {"spike", "short_burst", "bias"}:
        offsets[:] = direction * amplitude
    # dropout and stuck_sensor depend on observed readings, not random offsets
    return FaultPlan(event=event, offsets=offsets, direction=direction)


def apply_fault(plan: FaultPlan, current: float, *, position: int, previous_raw: float | None) -> float:
    """Apply one active transformation to a scalar sensor reading.

    Relative to a sensor value, gaussian/spike/bias/drift/noise/intermittent
    are additive. Corruption and stuck_sensor replace the current value;
    dropout emits NaN. Overlap precedence is managed by the batch injection engine.
    """
    event = plan.event
    if not (0 <= position < event.duration):
        raise ValueError("position must be within event")
    kind = event.fault_type
    if kind == "dropout":
        return float("nan")
    if kind == "stuck_sensor":
        if previous_raw is None:
            raise ValueError("stuck_sensor needs the previous raw reading")
        return float(previous_raw)
    if kind == "random_corruption":
        return float(plan.offsets[position])
    return float(current) + float(plan.offsets[position])
