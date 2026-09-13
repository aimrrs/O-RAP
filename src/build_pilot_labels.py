import requests
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

USERNAME = "aimrrs404@gmail.com"
PASSWORD = "qwertyuioplkjhgfdsa"

NORAD_ID = 25544

HORIZONS_HOURS = [6, 12, 24, 48]

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

HISTORY_URL = (
    "https://www.space-track.org/basicspacedata/query/"
    "class/gp_history/"
    f"NORAD_CAT_ID/{NORAD_ID}/"
    "EPOCH/%3E2025-01-01/"
    "orderby/CREATION_DATE%20asc/"
    "limit/100/"
    "format/json"
)

# --------------------------------------------------
# Login
# --------------------------------------------------

session = requests.Session()

login_response = session.post(
    LOGIN_URL,
    data={
        "identity": USERNAME,
        "password": PASSWORD,
    },
)

print("Login status:", login_response.status_code)

if login_response.status_code != 200:
    raise RuntimeError("Space-Track login failed.")

# --------------------------------------------------
# Download historical TLEs
# --------------------------------------------------

response = session.get(HISTORY_URL)

print("History status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError("Failed to download GP history.")

records = response.json()

print("Records returned:", len(records))

# --------------------------------------------------
# Convert to dataframe
# --------------------------------------------------

df = pd.DataFrame(records)

df["CREATION_DATE"] = pd.to_datetime(
    df["CREATION_DATE"],
    utc=True,
)

df["EPOCH"] = pd.to_datetime(
    df["EPOCH"],
    utc=True,
)

df = df.sort_values("CREATION_DATE").reset_index(drop=True)

# Remove exact duplicate TLEs
df = df.drop_duplicates(
    subset=["TLE_LINE1", "TLE_LINE2"],
    keep="first",
).reset_index(drop=True)

print("Unique TLEs:", len(df))

# --------------------------------------------------
# SGP4 helper
# --------------------------------------------------

def propagate(tle1, tle2, target_time):
    satellite = Satrec.twoline2rv(tle1, tle2)

    year = target_time.year
    month = target_time.month
    day = target_time.day

    hour = target_time.hour
    minute = target_time.minute
    second = (
        target_time.second
        + target_time.microsecond / 1_000_000
    )

    jd, fr = jday(
        year,
        month,
        day,
        hour,
        minute,
        second,
    )

    error, position, velocity = satellite.sgp4(jd, fr)

    if error != 0:
        return None

    return np.array(position)


# --------------------------------------------------
# Build pilot labels
# --------------------------------------------------

rows = []

for i, source in df.iterrows():

    source_time = source["CREATION_DATE"]

    for horizon in HORIZONS_HOURS:

        target_time = source_time + timedelta(
            hours=horizon
        )

        # Find the first later TLE available
        # at or after the target time.
        candidates = df[
            df["CREATION_DATE"] >= target_time
        ]

        if candidates.empty:
            continue

        reference = candidates.iloc[0]
        reference_delay_hours = (
            reference["CREATION_DATE"] - target_time
        ).total_seconds() / 3600

        if reference_delay_hours > 6:
            continue

        predicted_position = propagate(
            source["TLE_LINE1"],
            source["TLE_LINE2"],
            target_time,
        )

        reference_position = propagate(
            reference["TLE_LINE1"],
            reference["TLE_LINE2"],
            target_time,
        )

        if (
            predicted_position is None
            or reference_position is None
        ):
            continue

        error_km = np.linalg.norm(
            predicted_position - reference_position
        )

        rows.append({
            "norad_id": NORAD_ID,
            "source_creation": source_time,
            "target_time": target_time,
            "horizon_hours": horizon,
            "reference_creation": reference["CREATION_DATE"],
            "source_epoch": source["EPOCH"],
            "reference_epoch": reference["EPOCH"],
            "source_tle_age_hours": (
                source_time - source["EPOCH"]
            ).total_seconds() / 3600,
            "reference_delay_hours": (
                reference["CREATION_DATE"] - target_time
            ).total_seconds() / 3600,
            "proxy_error_km": error_km,
        })

# --------------------------------------------------
# Save
# --------------------------------------------------

result = pd.DataFrame(rows)

output_path = (
    "data/processed/iss_pilot_labels.csv"
)

result.to_csv(
    output_path,
    index=False,
)

print()
print("========================================")
print("Pilot label generation complete")
print("========================================")
print("Samples:", len(result))
print("Output:", output_path)

if not result.empty:
    print()
    print("Error statistics:")
    print(result["proxy_error_km"].describe())