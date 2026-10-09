from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math

import numpy as np
from sgp4.api import Satrec, jday
from xgboost import XGBClassifier

from features import build_feature_array, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "orap_xgboost.json"
METADATA_FILE = MODEL_DIR / "orap_metadata.json"
ALLOWED_HORIZONS = (6.0, 12.0, 24.0, 48.0)


def _as_utc(value):
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("creation_time must include a timezone.")
    return dt.astimezone(timezone.utc)


def load_artifacts():
    model = XGBClassifier()
    model.load_model(MODEL_FILE)
    with METADATA_FILE.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("features") != FEATURE_NAMES:
        raise ValueError("Feature-order mismatch between model metadata and feature extractor.")
    return model, metadata


def propagate_sgp4(line1, line2, target_time):
    satellite = Satrec.twoline2rv(line1, line2)
    timestamp = _as_utc(target_time)
    jd, fr = jday(timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
                  timestamp.minute, timestamp.second + timestamp.microsecond / 1_000_000.0)
    error_code, position, velocity = satellite.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(f"SGP4 propagation failed with error code {error_code}.")
    return {
        "error_code": int(error_code),
        "frame": "TEME",
        "position_km": [float(v) for v in position],
        "velocity_km_s": [float(v) for v in velocity],
    }


def predict(line1, line2, creation_time, horizon_hours):
    """Exploratory demo inference; score is uncalibrated and not an operational probability."""
    creation = _as_utc(creation_time)
    horizon = float(horizon_hours)
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError("horizon_hours must be one of 6, 12, 24, or 48.")
    model, metadata = load_artifacts()
    feature_array = build_feature_array(line1, line2, horizon, creation)
    matrix = np.asarray(feature_array, dtype=float).reshape(1, -1)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Could not construct a valid model feature vector.")
    # The model's output is intentionally shown as a raw score; the saved Platt
    # artifact is not used because its transfer performance was not supported.
    model_score = float(model.predict_proba(matrix)[0, 1])
    target = creation + timedelta(hours=horizon)
    state = propagate_sgp4(line1, line2, target)
    features = dict(zip(FEATURE_NAMES, [float(v) for v in matrix[0]]))
    return {
        "model_score": model_score,
        "score_definition": "Uncalibrated model output; not a calibrated probability or operational decision.",
        "horizon_hours": horizon,
        "tle_age_hours": features["source_tle_age_hours"],
        "sgp4": state,
        "features": features,
        "model_version": metadata.get("model_version", "unknown"),
        "creation_time": creation.isoformat(),
        "target_time": target.isoformat(),
        "target_definition": "Later-TLE SGP4 agreement proxy; not independently measured position truth.",
    }
