import numpy as np
import pandas as pd
import pytest

from pipeline_functions import encode_oxygen_over_threshold


def test_encode_oxygen_over_threshold_maps_the_two_levels():
    source = pd.DataFrame(
        {
            "Oxygen over threshold": [
                -0.0445689713425745,
                22.437134398135598,
                np.float64("22.43713439813560128755"),
            ],
            "CMS air pressure": [1.0, 2.0, 3.0],
        }
    )

    result = encode_oxygen_over_threshold(source)

    assert list(result["Oxygen over threshold"]) == [0, 1, 1]
    assert list(result["CMS air pressure"]) == [1.0, 2.0, 3.0]
    assert list(source["Oxygen over threshold"]) == pytest.approx(
        [-0.0445689713425745, 22.437134398135598, 22.43713439813560128755]
    )


def test_encode_oxygen_over_threshold_rejects_any_other_value():
    source = pd.DataFrame({"Oxygen over threshold": [-0.0445689713425745, 0.5]})

    with pytest.raises(ValueError, match="0.5"):
        encode_oxygen_over_threshold(source)


def test_encode_oxygen_over_threshold_requires_the_column():
    with pytest.raises(ValueError, match="Oxygen over threshold"):
        encode_oxygen_over_threshold(pd.DataFrame({"CMS air pressure": [1.0]}))
