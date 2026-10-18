"""Batch-only multi-contamination CSV exporter. Run: python generate_contamination_levels.py"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from fault_injection import generate_contamination_dataset, compute_reference_stats

# Configuration: edit here, then execute from project root.
DATA_FOLDER = Path("PW_0.5h")
SOURCE_SPLIT = "test"   # validation for development, test for final evaluation
OUTPUT_FOLDER = Path("contamination_benchmarks")
TARGET_PERCENTAGES = (0, 0.5, 1, 2, 5, 10)  # % of rows with any *sensor modification*
REPLICATES = 2
RANDOM_SEED = 42

SPECIAL_SHARE = 0.075  # 7.5% of modified rows are benign Gaussian or active intermittent
GAUSSIAN_SHARE_OF_SPECIAL = 0.5  # split those modified rows ~50/50 across the two
OVERLAP_EVENT_FRACTION = 0.20  # nested overlap event for ~20% of ordinary primary events
SAME_SENSOR_OVERLAP_PROBABILITY = 0.20
MIN_GAP = 0
READING_WINDOWS = (5, 10, 30, 60)
DURATIONS = (3, 5, 10, 30, 60)
SEVERITIES_SD = (0.5, 1.0, 2.0, 3.0)
SENSOR_COLS = ("CMS air pressure", "Concentrazione ossigeno base", "Nitrogen pressure")


def main():
    if SOURCE_SPLIT not in ("validation", "test"):
        raise ValueError("SOURCE_SPLIT must be validation or test")
    train_path = DATA_FOLDER / "train.csv"
    source_path = DATA_FOLDER / ("val.csv" if SOURCE_SPLIT == "validation" else "test.csv")
    if not train_path.exists() or not source_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {source_path}")
    train = pd.read_csv(train_path)
    source = pd.read_csv(source_path)
    stats = compute_reference_stats(train, SENSOR_COLS)
    directory = OUTPUT_FOLDER / SOURCE_SPLIT
    directory.mkdir(parents=True, exist_ok=True)
    summary = []
    settings = {
        "source_split": SOURCE_SPLIT, "source_file": str(source_path),
        "training_reference_file": str(train_path),
        "target_percentages": TARGET_PERCENTAGES, "replicates": REPLICATES,
        "random_seed": RANDOM_SEED, "special_share": SPECIAL_SHARE,
        "gaussian_share_of_special": GAUSSIAN_SHARE_OF_SPECIAL,
        "overlap_event_fraction": OVERLAP_EVENT_FRACTION,
        "same_sensor_overlap_probability": SAME_SENSOR_OVERLAP_PROBABILITY,
        "min_gap": MIN_GAP, "reading_windows": READING_WINDOWS,
        "durations": DURATIONS, "severities_sd": SEVERITIES_SD,
        "contamination_definition": "Distinct rows whose final numeric sensor values differ from original",
        "anomaly_definition": "synthetic_anomaly=1 for active non-benign faults",
        "special_definition": "modified rows in Gaussian noise or active intermittent-fault phases",
    }
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))

    for percent in TARGET_PERCENTAGES:
        for replicate in range(1, REPLICATES+1):
            seed = int(np.random.SeedSequence([
                RANDOM_SEED, int(round(percent*1000)), replicate,
            ]).generate_state(1)[0])
            data, events, metrics = generate_contamination_dataset(
                source, reference_stats=stats, sensor_cols=SENSOR_COLS,
                target_percent=percent, special_share=SPECIAL_SHARE,
                gaussian_share_of_special=GAUSSIAN_SHARE_OF_SPECIAL,
                overlap_event_fraction=OVERLAP_EVENT_FRACTION,
                same_sensor_overlap_probability=SAME_SENSOR_OVERLAP_PROBABILITY,
                min_gap=MIN_GAP, reading_windows=READING_WINDOWS,
                durations=DURATIONS, severities_sd=SEVERITIES_SD, seed=seed,
            )
            label = str(percent).replace('.', 'p')
            stem = f"modified_{label}pct_rep{replicate:02d}"
            csv_path = directory / f"{stem}.csv"
            events_path = directory / f"{stem}_events.csv"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # pandas serializing intended NaN dropouts
                data.to_csv(csv_path, index=False)
                events.to_csv(events_path, index=False)
            summary.append({
                "dataset": stem, "source_split": SOURCE_SPLIT, "replicate": replicate,
                **metrics, "csv_path": str(csv_path), "events_path": str(events_path),
            })
            print(f"{stem}: {metrics['modified_rows']:,} modified rows "
                  f"({metrics['actual_modified_percent']:.2f}%), "
                  f"{metrics['actual_special_share_percent']:.1f}% special, "
                  f"{metrics['anomalous_rows']:,} positive labels, "
                  f"{metrics['overlapping_rows']:,} overlap rows")
    pd.DataFrame(summary).to_csv(directory / "contamination_summary.csv", index=False)
    print("Saved to", directory.resolve())


if __name__ == "__main__":
    main()
