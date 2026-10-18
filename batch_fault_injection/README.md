# Offline mixed-fault benchmark generator

This is a **single, batch-only Python package**, consolidating the fault injection engine and multiple-contamination-level CSV exporter. There are no live data, streaming, MQTT, Kafka, or `process_one` components.

## Quick start

Place your files as follows:

```text
project/
├── generate_contamination_levels.py
├── example_usage.py
├── requirements.txt
├── fault_injection/
│   ├── __init__.py
│   ├── events.py
│   ├── faults.py
│   ├── scheduler.py
│   ├── batch.py
│   └── generator.py
└── PW_0.5h/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Install dependencies with `python -m pip install -r requirements.txt` and run:

```bash
python generate_contamination_levels.py
```

Default settings generate **12 CSV datasets**: contamination levels of 0%, 0.5%, 1%, 2%, 5%, 10%, each with 2 replicates. These are based on `val.csv` (development); training statistics come only from `train.csv`. Change `SOURCE_SPLIT = "test"` only when ready for held-out final evaluation. Changing the split writes to a different folder.

### Three different percentages: important!

Because **benign Gaussian jitter is not an anomaly** and **intermittent faults are only active on alternate readings**, the requested contamination target is explicitly defined here as:

- `actual_modified_percent = 100 × number of distinct rows with >=1 changed sensor / total rows`
- `actual_special_share_percent = 100 × changed rows with active Gaussian or intermittent / changed rows`
- `actual_anomaly_percent = 100 × rows with synthetic_anomaly=1 / total rows`

The first target is exact up to integer rounding: requested 5% on 2,600 readings means 130 actually modified rows. The special quota is 7.5% of those modified rows by default (rounded to the nearest row). Since benign Gaussian-only rows have anomaly label **0**, `actual_anomaly_percent` will generally be smaller than `actual_modified_percent`.

Use `SPECIAL_SHARE=0.05` or `0.10` for exactly 5% or 10% of modified rows (subject to rounding). `GAUSSIAN_SHARE_OF_SPECIAL=0.5` splits the special quota approximately evenly by changed readings into benign Gaussian and the active phases of intermittent faults. For very small datasets/targets, integer rounding may allow one kind to receive zero rows.

**Important:** `synthetic_modified=1` is *not* necessarily a detection target. Do not treat harmless jitter as an anomalous positive example. Use `synthetic_anomaly` for the known injected-fault label, and consider the original `PW_0.5h` future-failure target separately.

### Overlaps

Set `OVERLAP_EVENT_FRACTION=0.20` to add second nested events to approximately 20% of ordinary scheduled events (at least one if ordinary events exist). The nested event begins and ends inside an already modified interval so it **does not increase the target count of distinct modified rows**. `SAME_SENSOR_OVERLAP_PROBABILITY=0.20` controls how often secondary events share the parent's sensor; most overlaps affect different sensors. The generator reserves Gaussian/intermittent events from these automatic overlaps to keep the 5–10% special quota interpretable.

The engine supports manually supplied combinations, including same-sensor additive overlaps and different-sensor simultaneous faults. Same-sensor precedence: additive faults combine; `random_corruption` or `stuck_sensor` replaces them; dropout produces NaN last. **Overlapping replacements on the same sensor are rejected.** `overlap_count` counts all scheduled events at each timestamp, including off-phases of intermittent faults.

### Metadata and labels

Each generated CSV contains the original readings plus:

| Column | Meaning |
|---|---|
| `synthetic_modified` | Any sensor differs numerically from the original row (including benign noise) |
| `synthetic_anomaly` | One or more *intended true fault* phases are active |
| `injection_interval` | One or more event windows are active |
| `special_fault_active` | Gaussian jitter or an active intermittent fault occurs |
| `benign_noise_active` | Gaussian noise scheduled on the row |
| `intermittent_active` | Positive phase of an intermittent fault |
| `overlap_count` | Number of scheduled event windows covering this row |
| `synthetic_event_id`, `anomaly_id` | Pipe-separated active/event IDs; empty string if none |
| `synthetic_event_ids`, `anomaly_ids` | JSON arrays for robust parsing of multiple IDs |

Each dataset also has `_events.csv` describing each event, plus a `contamination_summary.csv` and `settings.json` shared across replicates. **Totals in event-level manifests must not be added together to estimate distinct changed rows when overlaps exist.** Use `synthetic_modified.sum()` instead.

### Why is `PW_0.5h` preserved?

`PW_0.5h = 1` means the machine **will fail within the next 30 minutes**, not that it is necessarily currently faulty. Injection candidate intervals (and their maximum reading-window history) require `PW_0.5h=0` and contiguous one-minute timestamps, but this does not certify normal machine operation. `PW_0.5h` is never injected or treated as a sensor feature. Label 0 is an exclusion heuristic, not verified ground truth of sensor health.

### Advanced use

To generate only one dataset:

```python
from fault_injection import compute_reference_stats, generate_contamination_dataset

stats = compute_reference_stats(train_df, sensor_cols)
combined, manifest, summary = generate_contamination_dataset(
    val_df, reference_stats=stats, sensor_cols=sensor_cols,
    target_percent=2, special_share=0.10, overlap_event_fraction=0.30,
    seed=42,
)
combined.to_csv("mixed_2pct.csv", index=False)
manifest.to_csv("mixed_2pct_events.csv", index=False)
```

For a custom event schedule, use `FaultEvent` and `inject_batch` (or the compatible batch-only `FaultEngine` wrapper). Note that `FaultEvent.start` and `duration` use **row positions**, not timestamps. The `schedule_random` helper can place manually specified event configurations into label-zero periods.

### Validation and caveats

Run `python -m unittest discover -s tests -v`. Tests cover all ten fault types, overlap priority, data preservation, deterministic generation, and six modified-row contamination levels.

- Keep `train.csv` unmodified when fitting a baseline normal-behaviour Isolation Forest; use synthesized `val.csv` copies for development and `test.csv` copies for final held-out evaluation.
- Missing readings from dropout are represented as NaN; later model pipelines will need a training-derived missing-value strategy.
- The generator enforces the modified-row budget and special share (rounded); it does **not** guarantee equally frequent event *types* or fixed proportions of benign versus true anomalies.
- Because source sensors may be quantized or slowly changing, a stuck value may sometimes equal real readings. This generator selects stuck events only when the full affected interval is detectably changed.
- The code tests **sensor faults**, not true mechanical failure forecasting.
