from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.inference import (
    COLUMNS,
    FailureModel,
    RandomForestModel,
    lagged_features,
)
from app.payload import combine
from app.readings import Reading
from binary_classification.pipeline_functions import (
    encode_oxygen_over_threshold,
    lag_window,
)

TRAIN_PATH = Path(__file__).resolve().parents[1] / "PW_0.5h" / "train.csv"
OXYGEN_LOW = -0.0445689713425745


def _rows(count: int) -> list[dict]:
    stamps = pd.date_range("2023-08-01 03:00:00", periods=count, freq="min")
    rows = []
    for stamp in stamps:
        rows.append(
            {
                "timestamp": str(stamp),
                "CMS air pressure": -0.5,
                "Concentrazione ossigeno base": -0.9,
                "Nitrogen pressure": -1.3,
                "Oxygen over threshold": OXYGEN_LOW,
                "PW_0.5h": 0,
            }
        )
    return rows


def _train_rows(count: int) -> list[dict]:
    frame = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"]).head(count)
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                "timestamp": str(record["timestamp"]),
                "CMS air pressure": float(record["CMS air pressure"]),
                "Concentrazione ossigeno base": float(record["Concentrazione ossigeno base"]),
                "Nitrogen pressure": float(record["Nitrogen pressure"]),
                "Oxygen over threshold": float(record["Oxygen over threshold"]),
                "PW_0.5h": int(record["PW_0.5h"]),
            }
        )
    return rows


def _reading() -> Reading:
    return Reading(
        index=29,
        total=100,
        timestamp="2023-08-01 03:29:00",
        cms_air_pressure=-0.5,
        oxygen_concentration=-0.9,
        nitrogen_pressure=-1.3,
        oxygen_over_threshold=OXYGEN_LOW,
        pw_0_5h=0,
        window_rows=30,
        window_required=30,
        rows=[],
    )


def test_lagged_features_matches_training_pipeline():
    rows = _rows(3)
    features = lagged_features(rows, 2)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp")
    expected = lag_window(encode_oxygen_over_threshold(frame), 2).drop(columns=["PW_0.5h"])

    pd.testing.assert_frame_equal(features, expected)
    oxygen = features.filter(like="Oxygen over threshold")
    assert set(oxygen.to_numpy().ravel().astype(int)) == {0}
    assert "PW_0.5h" not in features.columns


def test_failure_model_predicts_from_lagged_window():
    result = FailureModel().predict(_train_rows(30))

    assert result["threshold"] == 0.22
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["failure_predicted"] is (
        result["failure_probability"] >= result["threshold"]
    )


def test_random_forest_model_uses_saved_threshold(tmp_path):
    rows = _rows(30)
    features = lagged_features(rows, 30)
    training = pd.concat([features, features], ignore_index=True)
    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)
    model.fit(training, [0, 1])
    path = tmp_path / "forest.joblib"
    joblib.dump({"model": model, "threshold": 0.34}, path)

    result = RandomForestModel(path).predict(rows)

    assert result["forest_threshold"] == 0.34
    assert 0.0 <= result["forest_failure_probability"] <= 1.0
    assert result["forest_failure_predicted"] is (
        result["forest_failure_probability"] >= 0.34
    )


def test_random_forest_model_rejects_a_short_window(tmp_path):
    rows = _rows(30)
    features = lagged_features(rows, 30)
    training = pd.concat([features, features], ignore_index=True)
    model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0)
    model.fit(training, [0, 1])
    path = tmp_path / "forest.joblib"
    joblib.dump({"model": model, "threshold": 0.34}, path)

    with pytest.raises(ValueError, match="30"):
        RandomForestModel(path).predict(_rows(2))


def test_saved_random_forest_scores_a_training_window():
    result = RandomForestModel().predict(_train_rows(30))

    assert result["forest_threshold"] == 0.34
    assert 0.0 <= result["forest_failure_probability"] <= 1.0


def test_combine_includes_forest_prediction():
    payload = combine(
        _reading(),
        {"failure_probability": 0.1, "threshold": 0.22, "failure_predicted": False},
        None,
        20,
        {
            "forest_failure_probability": 0.4,
            "forest_threshold": 0.34,
            "forest_failure_predicted": True,
        },
    )

    assert payload["status"] == "ready"
    assert payload["forest_failure_probability"] == 0.4
    assert payload["forest_failure_predicted"] is True
    assert payload["survival_status"] == "more_data_required"


def test_combine_leaves_forest_empty_before_the_window_fills():
    payload = combine(_reading(), None, None, 20)

    assert payload["status"] == "too_small_window"
    assert payload["forest_failure_probability"] is None
    assert payload["forest_threshold"] is None
    assert payload["forest_failure_predicted"] is None
