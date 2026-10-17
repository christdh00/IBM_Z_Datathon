import numpy as np
import pandas as pd

OXYGEN_OVER_THRESHOLD = "Oxygen over threshold"
# The stored high value is one float64 step above this decimal.
_OXYGEN_LEVELS = (
    (-0.0445689713425745, 0),
    (22.437134398135598, 1),
)
_OXYGEN_MATCH_ATOL = 1e-12


def encode_oxygen_over_threshold(dataframe):
    """Map ``Oxygen over threshold`` onto the binary levels 0 and 1.

    ``-0.0445689713425745`` becomes 0 and ``22.437134398135598`` becomes 1.
    Any other value cancels the transform.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A frame that includes ``Oxygen over threshold``. Other columns are copied
        through unchanged.

    Returns
    -------
    pandas.DataFrame
        A copy of ``dataframe`` whose oxygen column is int 0 or 1.

    Raises
    ------
    ValueError
        The oxygen column is missing, or it contains a value that is neither
        of the two levels.
    """
    if OXYGEN_OVER_THRESHOLD not in dataframe.columns:
        raise ValueError(f"dataframe must include {OXYGEN_OVER_THRESHOLD!r}")

    values = dataframe[OXYGEN_OVER_THRESHOLD].to_numpy(dtype=float)
    matched = np.zeros(len(dataframe), dtype=bool)
    encoded_values = np.zeros(len(dataframe), dtype=np.int8)
    for level, code in _OXYGEN_LEVELS:
        is_level = np.isclose(values, level, rtol=0.0, atol=_OXYGEN_MATCH_ATOL)
        is_level &= np.isfinite(values)
        encoded_values[is_level] = code
        matched |= is_level

    if not matched.all():
        unexpected = np.unique(values[~matched])
        raise ValueError(
            f"{OXYGEN_OVER_THRESHOLD} has values other than the two expected levels: "
            f"{unexpected.tolist()}"
        )

    encoded = dataframe.copy()
    encoded[OXYGEN_OVER_THRESHOLD] = encoded_values
    return encoded


def lag_window(dataframe, rw):
    """Flatten a sliding reading window into lag columns.

    Each output row holds ``rw`` consecutive input rows, and the next row
    starts one input row later. Sensor columns are lagged. ``PW_0.5h``, when
    present, is kept as the label of the lag-0 row in each window.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        ``n`` rows and ``p`` columns. ``timestamp`` must be the index, or a
        column that can be set as the index. Rows are taken in their current
        order, oldest first.
    rw : int
        Window length. Each output row contains ``rw`` consecutive input rows.

    Returns
    -------
    pandas.DataFrame
        ``n - rw + 1`` rows when ``n >= rw``, otherwise no rows. Each feature
        column becomes ``rw`` lag columns, grouped by lag. Within lag ``k``
        the columns keep the original sensor order, and lag ``0`` comes before
        lag ``1``. Column ``{name}_lag_0`` is the newest value in the window;
        ``{name}_lag_{rw-1}`` is the oldest. The index is that newest row's
        timestamp.

        The window advances by one row, so ``{name}_lag_k`` on one row equals
        ``{name}_lag_{k+1}`` on the next row.

        ``PW_0.5h`` is the future-failure label, so it is not lagged. Each
        output row keeps the ``PW_0.5h`` value from its lag-0 row.
    """
    label_name = "PW_0.5h"
    rw = int(rw)
    if rw < 1:
        raise ValueError("rw must be a positive integer")

    frame = dataframe
    if frame.index.name != "timestamp":
        if "timestamp" not in frame.columns:
            raise ValueError('dataframe must have an index or column named "timestamp"')
        frame = frame.set_index("timestamp")

    labels = frame[label_name] if label_name in frame.columns else None
    features = frame.drop(columns=[label_name]) if labels is not None else frame

    n_rows, n_cols = features.shape
    n_windows = n_rows - rw + 1
    columns = [
        f"{name}_lag_{lag}"
        for lag in range(rw)
        for name in features.columns
    ]

    if n_windows <= 0:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "timestamp"
        if labels is not None:
            empty[label_name] = pd.Series(dtype=labels.dtype)
        return empty

    # (window, feature, time oldest→newest)
    windows = np.lib.stride_tricks.sliding_window_view(
        features.to_numpy(),
        rw,
        axis=0,
    )
    # (window, lag newest→oldest, feature) so columns that share a lag stay together
    by_lag = np.ascontiguousarray(windows[:, :, ::-1].transpose(0, 2, 1))
    flat = by_lag.reshape(n_windows, rw * n_cols)

    result = pd.DataFrame(
        flat,
        index=features.index[rw - 1 :],
        columns=columns,
    )
    result.index.name = "timestamp"
    if labels is not None:
        result[label_name] = labels.iloc[rw - 1 :].to_numpy()
    return result


def undo_scale_pos_weight(proba, scale_pos_weight):
    """Map class-weighted probabilities back toward the original prevalence.

    ``scale_pos_weight`` multiplies the positive class in the training loss.
    That is the same as training where each positive row is repeated
    ``scale_pos_weight`` times, so a fitted probability ``p_w`` satisfies

        p_w / (1 - p_w) = scale_pos_weight * p / (1 - p).

    The probability on the original prevalence is

        p = p_w / (p_w + scale_pos_weight * (1 - p_w)).

    Parameters
    ----------
    proba : array-like
        Predicted probabilities from a model trained with ``scale_pos_weight``.
        Every value must be finite and lie in ``[0, 1]``.
    scale_pos_weight : float
        Positive class weight used in training. Must be finite and positive.
        A weight of ``1`` leaves the probabilities unchanged.

    Returns
    -------
    numpy.ndarray
        Probabilities on the unweighted prevalence, with the same shape as
        ``proba``.

    Raises
    ------
    ValueError
        ``scale_pos_weight`` is not a positive finite number, or a probability
        is non-finite or outside ``[0, 1]``.
    """
    weighted = np.asarray(proba, dtype=float)
    weight = float(scale_pos_weight)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("scale_pos_weight must be a positive finite number")
    if weighted.size and (
        not np.isfinite(weighted).all() or weighted.min() < 0 or weighted.max() > 1
    ):
        raise ValueError("proba values must be finite and lie in [0, 1]")
    return weighted / (weighted + weight * (1.0 - weighted))
