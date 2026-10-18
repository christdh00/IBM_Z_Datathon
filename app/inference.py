import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from anom_dectection.anom_detection_isolation_forest import extract_features
from binary_classification.pipeline_functions import (
    encode_oxygen_over_threshold,
    lag_window,
    undo_scale_pos_weight,
)
from survival.survival_transform import (
    HORIZONS,
    prepare_live_features,
    trustguard_output,
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
FOREST_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "binary_classification"
    / "models"
    / "random_forest_model_v1.joblib"
)


def lagged_features(rows: list[dict], window_minutes: int) -> pd.DataFrame:
    """Lag one reading window the same way the binary models were trained."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp")
    lagged = lag_window(encode_oxygen_over_threshold(frame), window_minutes)
    return lagged.drop(columns=["PW_0.5h"])


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
        features = lagged_features(rows, self.window_minutes)
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


class RandomForestModel:
    """Predict a failure probability with the scikit-learn random forest."""

    def __init__(self, model_path: Path = FOREST_MODEL_PATH) -> None:
        self.window_minutes = WINDOW_MINUTES
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.threshold = float(saved["threshold"])

    def predict(self, rows: list[dict]) -> dict:
        if len(rows) != self.window_minutes:
            raise ValueError(
                f"model expects {self.window_minutes} rows, received {len(rows)}"
            )
        features = lagged_features(rows, self.window_minutes)
        probability = float(self.model.predict_proba(features)[:, 1][0])
        return {
            "forest_failure_probability": probability,
            "forest_threshold": self.threshold,
            "forest_failure_predicted": probability >= self.threshold,
        }


SURVIVAL_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "survival"
    / "models"
    / "xgboost_aft_model_v1.json"
)


class SurvivalModel:
    """Predict time-to-failure from the latest 20 sensor readings."""

    def __init__(self, model_path: Path = SURVIVAL_MODEL_PATH) -> None:
        self.window_minutes = 20
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        best_iteration = self.model.attr("best_iteration")
        self.iteration_range = (
            (0, int(best_iteration) + 1) if best_iteration is not None else None
        )
        config = json.loads(self.model.save_config())
        loss = config["learner"]["objective"]["aft_loss_param"]
        self.scale = float(loss["aft_loss_distribution_scale"])

    def predict(self, rows: list[dict]) -> dict:
        if len(rows) < self.window_minutes:
            raise ValueError(
                f"model expects at least {self.window_minutes} rows, received {len(rows)}"
            )
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        features = prepare_live_features(frame, window=self.window_minutes)
        features = features.loc[:, self.model.feature_names]
        for name, kind in zip(self.model.feature_names, self.model.feature_types):
            if kind == "int":
                features[name] = features[name].astype(int)
        matrix = xgb.DMatrix(features)
        predict_kwargs = {"output_margin": True}
        if self.iteration_range is not None:
            predict_kwargs["iteration_range"] = self.iteration_range
        margin = float(self.model.predict(matrix, **predict_kwargs)[0])
        horizons = np.asarray(HORIZONS, dtype=float)
        failure = 1.0 - norm.sf((np.log(horizons) - margin) / self.scale)
        return trustguard_output(
            timestamp=str(frame["timestamp"].max()),
            failure_30=failure[0],
            failure_60=failure[1],
            failure_120=failure[2],
            median_time_to_failure_min=float(np.exp(margin)),
        )


ISO_WINDOW_MINUTES = 60
ISO_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "anom_dectection"
    / "iso_forest_nitrogen.pkl"
)
ISO_SCALER_PATH = (
    Path(__file__).resolve().parents[1] / "anom_dectection" / "iso_scaler.pkl"
)
ISO_SENSORS = [
    "CMS air pressure",
    "Concentrazione ossigeno base",
    "Nitrogen pressure",
    "Oxygen over threshold",
]
# Same limits the isolation forest notebook used when it fit the scaler.
ISO_RANGES = {
    "CMS air pressure": (0, 9.5),
    "Concentrazione ossigeno base": (5, 500),
    "Nitrogen pressure": (0, 9),
    "Oxygen over threshold": (0, 1),
}
ISO_JUMP_THRESHOLDS = {
    "CMS air pressure": 5,
    "Concentrazione ossigeno base": 1,
    "Nitrogen pressure": 0.2,
    "Oxygen over threshold": 0.1,
}
ISO_BASELINE_STATS = {
    "CMS air pressure": (1000, 10),
    "Concentrazione ossigeno base": (21, 1),
    "Nitrogen pressure": (2.0, 0.2),
    "Oxygen over threshold": (-0.045, 22.5),
}


def isolation_features(rows: list[dict]) -> pd.DataFrame:
    """Summarize one reading window the same way the isolation forest was trained."""
    sensors = pd.DataFrame(rows).loc[:, ISO_SENSORS]
    return extract_features(sensors, ISO_RANGES, ISO_JUMP_THRESHOLDS, ISO_BASELINE_STATS)


class IsolationForestModel:
    """Score one 60-minute reading window with the fitted isolation forest."""

    def __init__(
        self,
        model_path: Path = ISO_MODEL_PATH,
        scaler_path: Path = ISO_SCALER_PATH,
    ) -> None:
        self.window_minutes = ISO_WINDOW_MINUTES
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, rows: list[dict]) -> dict:
        if len(rows) != self.window_minutes:
            raise ValueError(
                f"model expects {self.window_minutes} rows, received {len(rows)}"
            )
        features = isolation_features(rows).loc[:, self.scaler.feature_names_in_]
        scaled = self.scaler.transform(features)
        trust_score = float(-self.model.score_samples(scaled)[0])
        anomaly = int(self.model.predict(scaled)[0]) == -1
        return {
            "isolation_trust_score": trust_score,
            "isolation_anomaly": anomaly,
        }
