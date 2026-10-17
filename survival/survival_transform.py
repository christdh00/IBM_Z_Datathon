import numpy as np
import pandas as pd

TARGET = "PW_0.5h"
SENSORS = ["CMS air pressure", "Concentrazione ossigeno base", "Nitrogen pressure", "Oxygen over threshold"]
BINARY_SENSOR = "Oxygen over threshold"
WINDOWS = 20
STEP = 5

# Create survival targets
def create_survival_targets(data, target=TARGET):
    data = data.sort_values("timestamp").reset_index(drop=True).copy()
    changes = data[target].diff().fillna(0)
    failure_rows = changes == -1
    failure_times = data.loc[failure_rows, "timestamp"].sort_values().to_numpy()
    timestamps = data["timestamp"].to_numpy()
    next_idx = np.searchsorted(failure_times, timestamps)
    has_future_failure = next_idx < len(failure_times)
    data["event"] = has_future_failure.astype(int)
    data["next_failure_timestamp"] = pd.NaT
    data.loc[has_future_failure, "next_failure_timestamp"] = (failure_times[next_idx[has_future_failure]])
    data["time_to_event_min"] = np.nan
    data.loc[has_future_failure, "time_to_event_min"] = (
        data.loc[has_future_failure, "next_failure_timestamp"] - data.loc[has_future_failure, "timestamp"]
    ).dt.total_seconds() / 60
    ending = data["timestamp"].max()
    data.loc[~has_future_failure, "time_to_event_min"] = (ending - data.loc[~has_future_failure, 
        "timestamp"]).dt.total_seconds() / 60
    data["episode_id"] = failure_rows.cumsum().shift(fill_value=0).astype(int)
    return data[data["time_to_event_min"] > 0].copy()

# Extracteing sensor features
def extract_sensor_features(x, sensors=SENSORS):
    row = {}

    for sensor in sensors:
        values = x[sensor].to_numpy()

        if sensor == BINARY_SENSOR:
            row[f"{sensor}_current"] = values[-1]
            row[f"{sensor}_proportion"] = values.mean()
            row[f"{sensor}_transitions"] = np.sum(
                np.diff(values) != 0
            )

        else:
            row[f"{sensor}_current"] = values[-1]
            row[f"{sensor}_mean"] = values.mean()
            row[f"{sensor}_std"] = values.std()
            row[f"{sensor}_slope"] = np.polyfit(
                np.arange(len(values)), values, 1
            )[0]

    return row

# Create rolling features
def create_rolling_features(data, sensors=SENSORS, window=WINDOWS, step=STEP):
    rows = []
    for episode_id, episode in data.groupby("episode_id"):
        episode = episode.sort_values("timestamp").reset_index(drop=True)
        for end in range(window - 1, len(episode), step):
            x = episode.iloc[end - window + 1:end + 1]
            current = x.iloc[-1]
            row = {
                "episode_id": episode_id,
                "timestamp": current["timestamp"],
                "time_to_event_min": current["time_to_event_min"],
                "event": current["event"]
            }
            row.update(extract_sensor_features(x, sensors))
            rows.append(row)
    return pd.DataFrame(rows)

# Create live features
def create_life_features(data, sensors=SENSORS, window=WINDOWS):
    data = data.sort_values("timestamp").tail(window).copy()
    if len(data) < window:
        raise ValueError(f"Need {window} readings, but only received {len(data)}.")
    row = extract_sensor_features(data, sensors)
    return pd.DataFrame([row])