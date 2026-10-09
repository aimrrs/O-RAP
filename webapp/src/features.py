from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re

import numpy as np
from sgp4.api import Satrec

FEATURE_NAMES = [
    "horizon_hours", "source_tle_age_hours", "mean_motion", "eccentricity",
    "inclination", "ra_of_asc_node", "arg_of_pericenter", "mean_anomaly",
    "bstar", "mean_motion_dot", "mean_motion_ddot", "semimajor_axis",
    "period", "apoapsis", "periapsis",
]
EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418


def parse_tle(line1: str, line2: str) -> Satrec:
    if not isinstance(line1, str) or not isinstance(line2, str):
        raise TypeError("TLE lines must be strings.")
    line1, line2 = line1.strip(), line2.strip()
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("Expected standard TLE lines beginning with '1 ' and '2 '.")
    if len(line1) != 69 or len(line2) != 69:
        raise ValueError("Each TLE line must contain exactly 69 characters.")
    if line1[2:7] != line2[2:7]:
        raise ValueError("The satellite numbers in the TLE lines do not match.")
    return Satrec.twoline2rv(line1, line2)


def _parse_tle_decimal_field(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    match = re.fullmatch(r"([+-]?)(\d+)([+-]\d+)", value)
    if not match:
        raise ValueError(f"Invalid TLE implied-decimal field: {value!r}")
    sign, mantissa, exponent = match.groups()
    return float(f"{sign}0.{mantissa}") * 10.0 ** int(exponent)


def _parse_tle_epoch(line1: str) -> datetime:
    """Read the UTC epoch from TLE line 1 columns 19–32 (YYDDD.DDD...)."""
    field = line1[18:32]
    try:
        yy = int(field[:2])
        day_of_year = float(field[2:])
    except (ValueError, IndexError) as exc:
        raise ValueError("TLE line 1 contains an invalid epoch field.") from exc
    year = 1900 + yy if yy >= 57 else 2000 + yy
    if not (1.0 <= day_of_year < 367.0):
        raise ValueError("TLE epoch day-of-year is outside its valid range.")
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def _parse_datetime(value) -> datetime:
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


def calculate_orbital_features(sat: Satrec, line1: str, line2: str) -> dict[str, float]:
    mean_motion = sat.no_kozai * 1440.0 / (2.0 * math.pi)
    eccentricity = float(sat.ecco)
    inclination = math.degrees(float(sat.inclo))
    ra_of_asc_node = math.degrees(float(sat.nodeo))
    arg_of_pericenter = math.degrees(float(sat.argpo))
    mean_anomaly = math.degrees(float(sat.mo))
    bstar = float(sat.bstar)
    mean_motion_dot = float(line1[33:43].strip() or 0.0)
    mean_motion_ddot = _parse_tle_decimal_field(line1[44:52])
    n_rad_per_sec = float(sat.no_kozai) / 60.0
    if n_rad_per_sec <= 0:
        raise ValueError("Mean motion must be positive.")
    semimajor_axis = (EARTH_MU_KM3_S2 / n_rad_per_sec**2) ** (1.0 / 3.0)
    period_minutes = 2.0 * math.pi / float(sat.no_kozai)
    return {
        "mean_motion": mean_motion, "eccentricity": eccentricity,
        "inclination": inclination, "ra_of_asc_node": ra_of_asc_node,
        "arg_of_pericenter": arg_of_pericenter, "mean_anomaly": mean_anomaly,
        "bstar": bstar, "mean_motion_dot": mean_motion_dot,
        "mean_motion_ddot": mean_motion_ddot, "semimajor_axis": semimajor_axis,
        "period": period_minutes,
        "apoapsis": semimajor_axis * (1.0 + eccentricity) - EARTH_RADIUS_KM,
        "periapsis": semimajor_axis * (1.0 - eccentricity) - EARTH_RADIUS_KM,
    }


def calculate_tle_age_hours(line1: str, creation_time) -> float:
    """Research feature: source TLE creation timestamp minus its orbital epoch."""
    return (_parse_datetime(creation_time) - _parse_tle_epoch(line1)).total_seconds() / 3600.0


def build_features(line1: str, line2: str, horizon_hours: float, creation_time=None) -> dict[str, float]:
    horizon_hours = float(horizon_hours)
    if not math.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError("horizon_hours must be finite and greater than zero.")
    if creation_time is None:
        raise ValueError("creation_time is required to calculate source TLE age.")
    sat = parse_tle(line1, line2)
    features = {
        "horizon_hours": horizon_hours,
        "source_tle_age_hours": calculate_tle_age_hours(line1, creation_time),
        **calculate_orbital_features(sat, line1, line2),
    }
    output = {}
    for name in FEATURE_NAMES:
        value = float(features[name])
        if not math.isfinite(value):
            raise ValueError(f"Feature {name} is not finite: {value}")
        output[name] = value
    return output


def build_feature_array(line1: str, line2: str, horizon_hours: float, creation_time=None) -> np.ndarray:
    values = build_features(line1, line2, horizon_hours, creation_time)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
