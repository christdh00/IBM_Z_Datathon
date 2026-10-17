import numpy as np
import pytest

from pipeline_functions import undo_scale_pos_weight


def test_undo_scale_pos_weight_divides_the_odds_by_the_weight():
    weighted = np.array([0.0, 0.5, 0.94, 1.0])
    weight = 31660 / 600

    corrected = undo_scale_pos_weight(weighted, weight)

    expected = weighted / (weighted + weight * (1.0 - weighted))
    assert corrected == pytest.approx(expected)
    assert corrected[1] == pytest.approx(0.5 / (0.5 * (1.0 + weight)))
    assert corrected[1] < 0.5
    assert corrected[0] == 0.0
    assert corrected[-1] == 1.0


def test_undo_scale_pos_weight_is_identity_when_the_weight_is_one():
    proba = np.array([0.0, 0.2, 0.8, 1.0])

    assert undo_scale_pos_weight(proba, 1) == pytest.approx(proba)


def test_undo_scale_pos_weight_preserves_order_and_does_not_mutate_input():
    proba = np.linspace(0.0, 1.0, 11)
    original = proba.copy()

    corrected = undo_scale_pos_weight(proba, 52.77)

    assert np.all(np.diff(corrected) > 0)
    assert original == pytest.approx(proba)


def test_undo_scale_pos_weight_rejects_a_non_positive_weight():
    with pytest.raises(ValueError, match="scale_pos_weight"):
        undo_scale_pos_weight([0.2], 0)
    with pytest.raises(ValueError, match="scale_pos_weight"):
        undo_scale_pos_weight([0.2], -3)


def test_undo_scale_pos_weight_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError, match="proba"):
        undo_scale_pos_weight([1.1], 2)
    with pytest.raises(ValueError, match="proba"):
        undo_scale_pos_weight([-0.01], 2)
    with pytest.raises(ValueError, match="proba"):
        undo_scale_pos_weight([np.nan], 2)
