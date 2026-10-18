"""Batch-only generation of reproducible synthetic sensor fault benchmarks."""
from .events import FaultEvent, FAULT_TYPES
from .faults import FaultPlan, prepare_fault, apply_fault
from .scheduler import eligible_starts, schedule_random, validate_events
from .batch import inject_batch, FaultEngine
from .generator import generate_contamination_dataset, compute_reference_stats

__all__ = [
    "FaultEvent", "FAULT_TYPES", "FaultPlan", "prepare_fault", "apply_fault",
    "eligible_starts", "schedule_random", "validate_events", "inject_batch", "FaultEngine",
    "generate_contamination_dataset", "compute_reference_stats",
]
