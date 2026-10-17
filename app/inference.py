from collections import deque
from pathlib import Path

import pandas as pd
import xgboost as xgb

from binary_classification.pipeline_functions import (
    encode_oxygen_over_threshold,
    lag_window,
    undo_scale_pos_weight,
)

WINDOW_MINUTES = 30
# Training fit rows: 32,260 rows, 600 positives. (32260 - 600) / 600.
SCALE_POS_WEIGHT = 31660 / 600
# F1 threshold chosen on the training eval holdout.
THRESHOLD = 0.22
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "binary_classification"
    / "models"
    / "xgboost_model_v1.json"
)
COLUMNS = [
    "timestamp",
    "CMS air pressure",
    "Concentrazione ossigeno base",
    "Nitrogen pressure",
    "Oxygen over threshold",
    "PW_0.5h",
]


class StreamScorer:
    """Score each streamed row once a 30-minute reading window is available."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        best_iteration = self.model.attr("best_iteration")
        self.iteration_range = (
            (0, int(best_iteration) + 1) if best_iteration is not None else None
        )
        self.rows: deque[dict] = deque(maxlen=WINDOW_MINUTES)
        self._last_index: int | None = None

    def score(self, streamed: dict) -> dict:
        index = streamed["index"]
        if self._last_index is not None and index <= self._last_index:
            self.rows.clear()
        self._last_index = index

        record = _record(streamed["record"])
        self.rows.append(record)
        payload = _public_row(streamed["index"], streamed["total"], record)
        payload["window_rows"] = len(self.rows)
        payload["window_required"] = WINDOW_MINUTES

        if len(self.rows) < WINDOW_MINUTES:
            payload["status"] = "too_small_window"
            payload["message"] = "too small window"
            payload["failure_probability"] = None
            payload["failure_predicted"] = None
            return payload

        probability = self._probability()
        payload["status"] = "ready"
        payload["message"] = None
        payload["failure_probability"] = probability
        payload["threshold"] = THRESHOLD
        payload["failure_predicted"] = probability >= THRESHOLD
        return payload

    def _probability(self) -> float:
        frame = pd.DataFrame(list(self.rows), columns=COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp")
        lagged = lag_window(encode_oxygen_over_threshold(frame), WINDOW_MINUTES)
        features = lagged.drop(columns=["PW_0.5h"])
        for column in features.columns:
            if column.startswith("Oxygen over threshold_lag_"):
                features[column] = pd.Categorical(
                    features[column].astype(int),
                    categories=[0, 1],
                )
        matrix = xgb.DMatrix(features, enable_categorical=True)
        predict_kwargs = {}
        if self.iteration_range is not None:
            predict_kwargs["iteration_range"] = self.iteration_range
        weighted = self.model.predict(matrix, **predict_kwargs)
        probability = undo_scale_pos_weight(weighted, SCALE_POS_WEIGHT)
        return float(probability[0])


def _record(row: dict) -> dict:
    return {
        "timestamp": str(row["timestamp"]),
        "CMS air pressure": float(row["CMS air pressure"]),
        "Concentrazione ossigeno base": float(row["Concentrazione ossigeno base"]),
        "Nitrogen pressure": float(row["Nitrogen pressure"]),
        "Oxygen over threshold": float(row["Oxygen over threshold"]),
        "PW_0.5h": int(row["PW_0.5h"]),
    }


def _public_row(index: int, total: int, record: dict) -> dict:
    return {
        "index": index,
        "total": total,
        "timestamp": record["timestamp"],
        "cms_air_pressure": record["CMS air pressure"],
        "oxygen_concentration": record["Concentrazione ossigeno base"],
        "nitrogen_pressure": record["Nitrogen pressure"],
        "oxygen_over_threshold": record["Oxygen over threshold"],
        "pw_0_5h": record["PW_0.5h"],
    }
