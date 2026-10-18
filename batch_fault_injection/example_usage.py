"""Minimal example: create one mixed-fault validation dataset from train/val CSVs."""
from pathlib import Path
import pandas as pd

from fault_injection import compute_reference_stats, generate_contamination_dataset

sensor_cols = ("CMS air pressure", "Concentrazione ossigeno base", "Nitrogen pressure")
train = pd.read_csv("PW_0.5h/train.csv")
validation = pd.read_csv("PW_0.5h/val.csv")
reference_stats = compute_reference_stats(train, sensor_cols)

combined, events, report = generate_contamination_dataset(
    validation, reference_stats=reference_stats, sensor_cols=sensor_cols,
    target_percent=5.0, special_share=0.075, overlap_event_fraction=0.30,
    seed=42,
)
Path("mixed_fault_outputs").mkdir(exist_ok=True)
combined.to_csv("mixed_fault_outputs/validation_5pct.csv", index=False)
events.to_csv("mixed_fault_outputs/validation_5pct_events.csv", index=False)
print(report)
