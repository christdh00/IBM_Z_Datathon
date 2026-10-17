import numpy as np
import pandas as pd


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
