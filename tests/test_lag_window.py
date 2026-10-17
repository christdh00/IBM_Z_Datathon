import pandas as pd
import pytest

from pipeline_functions import lag_window


def _sample_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2023-08-01 03:05:00",
                    "2023-08-01 03:06:00",
                    "2023-08-01 03:07:00",
                    "2023-08-01 03:08:00",
                    "2023-08-01 03:09:00",
                ]
            ),
            "CMS air pressure": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Nitrogen pressure": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).set_index("timestamp")


def test_lag_window_prints_transformed_frame():
    source = _sample_frame()
    rw = 2
    result = lag_window(source, rw)

    print("input")
    print(source.to_string())
    print("\nlag_window(dataframe, rw=2)")
    print(result.to_string())

    assert result.shape == (4, 4)
    assert list(result.columns) == [
        "CMS air pressure_lag_0",
        "Nitrogen pressure_lag_0",
        "CMS air pressure_lag_1",
        "Nitrogen pressure_lag_1",
    ]
    assert list(result.index) == list(source.index[1:])
    assert result.index.name == "timestamp"

    assert list(result["CMS air pressure_lag_0"]) == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert list(result["CMS air pressure_lag_1"]) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert list(result["Nitrogen pressure_lag_0"]) == pytest.approx([20.0, 30.0, 40.0, 50.0])
    assert list(result["Nitrogen pressure_lag_1"]) == pytest.approx([10.0, 20.0, 30.0, 40.0])

    # lag 0 of one row is lag 1 of the next row
    assert list(result["CMS air pressure_lag_0"].iloc[:-1]) == pytest.approx(
        list(result["CMS air pressure_lag_1"].iloc[1:])
    )

    wider = lag_window(source, rw=3)
    assert len(wider) == len(source) - 3 + 1
    air = source["CMS air pressure"].to_numpy()
    assert list(wider["CMS air pressure_lag_0"]) == pytest.approx(list(air[2:]))
    assert list(wider["CMS air pressure_lag_1"]) == pytest.approx(list(air[1:-1]))
    assert list(wider["CMS air pressure_lag_2"]) == pytest.approx(list(air[:-2]))


def test_lag_window_labels_each_row_with_lag_0_pw():
    source = _sample_frame()
    # Differs inside each window so the label cannot be taken from an older lag.
    source["PW_0.5h"] = [0, 1, 0, 1, 0]

    result = lag_window(source, rw=2)

    assert "PW_0.5h" in result.columns
    assert not any(column.startswith("PW_0.5h_lag_") for column in result.columns)
    assert list(result["PW_0.5h"]) == [1, 0, 1, 0]
    assert list(result["PW_0.5h"]) == list(source.loc[result.index, "PW_0.5h"])
