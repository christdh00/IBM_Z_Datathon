from app.readings import Reading

SURVIVAL_FIELDS = (
    "survival_model",
    "failure_probability_30_min",
    "failure_probability_60_min",
    "failure_probability_120_min",
    "median_time_to_failure_min",
)


FOREST_FIELDS = (
    "forest_failure_probability",
    "forest_threshold",
    "forest_failure_predicted",
)


def combine(
    reading: Reading,
    failure: dict | None,
    survival: dict | None,
    survival_window_minutes: int,
    forest: dict | None = None,
) -> dict:
    """Join the current reading with each model's prediction."""
    payload = {
        "index": reading.index,
        "total": reading.total,
        "timestamp": reading.timestamp,
        "cms_air_pressure": reading.cms_air_pressure,
        "oxygen_concentration": reading.oxygen_concentration,
        "nitrogen_pressure": reading.nitrogen_pressure,
        "oxygen_over_threshold": reading.oxygen_over_threshold,
        "pw_0_5h": reading.pw_0_5h,
        "window_rows": reading.window_rows,
        "window_required": reading.window_required,
        "survival_window_required": survival_window_minutes,
    }
    if failure is None:
        payload["status"] = "too_small_window"
        payload["message"] = "too small window"
        payload["failure_probability"] = None
        payload["failure_predicted"] = None
    else:
        payload["status"] = "ready"
        payload["message"] = None
        payload.update(failure)

    if forest is None:
        for field in FOREST_FIELDS:
            payload[field] = None
    else:
        for field in FOREST_FIELDS:
            payload[field] = forest[field]

    if survival is None:
        payload["survival_status"] = "more_data_required"
        payload["survival_message"] = "more data required"
        for field in SURVIVAL_FIELDS:
            payload[field] = None
        return payload

    payload["survival_status"] = "ready"
    payload["survival_message"] = None
    for field in SURVIVAL_FIELDS:
        payload[field] = survival[field]
    return payload
