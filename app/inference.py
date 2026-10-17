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


class FailureModel:
    """Predict a failure probability from one 30-minute reading window."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.window_minutes = WINDOW_MINUTES
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        best_iteration = self.model.attr("best_iteration")
        self.iteration_range = (
            (0, int(best_iteration) + 1) if best_iteration is not None else None
        )

    def predict(self, rows: list[dict]) -> dict:
        if len(rows) != self.window_minutes:
            raise ValueError(
                f"model expects {self.window_minutes} rows, received {len(rows)}"
            )
        weighted = self._weighted_probability(rows)
        probability = float(undo_scale_pos_weight(weighted, SCALE_POS_WEIGHT)[0])
        return {
            "failure_probability": probability,
            "threshold": THRESHOLD,
            "failure_predicted": probability >= THRESHOLD,
        }

    def _weighted_probability(self, rows: list[dict]):
        frame = pd.DataFrame(rows, columns=COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp")
        lagged = lag_window(encode_oxygen_over_threshold(frame), self.window_minutes)
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
        return self.model.predict(matrix, **predict_kwargs)
