from app.readings import Reading


def combine(reading: Reading, prediction: dict | None) -> dict:
    """Join the current reading with a model prediction into one page payload."""
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
    }
    if prediction is None:
        payload["status"] = "too_small_window"
        payload["message"] = "too small window"
        payload["failure_probability"] = None
        payload["failure_predicted"] = None
        return payload

    payload["status"] = "ready"
    payload["message"] = None
    payload.update(prediction)
    return payload
