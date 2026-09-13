import requests
import pandas as pd
import numpy as np

from sgp4.api import Satrec, jday


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NORAD_ID = 25544

USERNAME = "aimrrs404@gmail.com"
PASSWORD = "qwertyuioplkjhgfdsa"

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

login = session.post(
    "https://www.space-track.org/ajaxauth/login",
    data={
        "identity": USERNAME,
        "password": PASSWORD,
    },
)

print("Login status:", login.status_code)

if login.status_code != 200:
    raise RuntimeError("Space-Track login failed.")


# --------------------------------------------------
# Download TLE history
# --------------------------------------------------

response = session.get(HISTORY_URL)

print("History status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError("Failed to download TLE history.")

records = response.json()

df = pd.DataFrame(records)

df["CREATION_DATE"] = pd.to_datetime(
    df["CREATION_DATE"],
    utc=True,
)

df = (
    df.drop_duplicates(
        subset=["TLE_LINE1", "TLE_LINE2"],
        keep="first",
    )
    .sort_values("CREATION_DATE")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Load pilot labels
# --------------------------------------------------

labels = pd.read_csv(
    "data/processed/iss_pilot_labels.csv"
)

labels["source_creation"] = pd.to_datetime(
    labels["source_creation"],
    utc=True,
)

labels["reference_creation"] = pd.to_datetime(
    labels["reference_creation"],
    utc=True,
)

labels["target_time"] = pd.to_datetime(
    labels["target_time"],
    utc=True,
)


# --------------------------------------------------
# Select 20 representative samples
# --------------------------------------------------

selected = []

for horizon, group in labels.groupby("horizon_hours"):

    group = group.sort_values("proxy_error_km")

    indices = [
        0,
        len(group) // 4,
        len(group) // 2,
        3 * len(group) // 4,
        len(group) - 1,
    ]

    for index in indices:
        selected.append(group.iloc[index])

selected = (
    pd.DataFrame(selected)
    .drop_duplicates()
    .reset_index(drop=True)
)

print()
print("Selected samples:", len(selected))


# --------------------------------------------------
# TLE lookup
# --------------------------------------------------

tle_lookup = df.set_index("CREATION_DATE")


def propagate(tle1, tle2, target_time):

    satellite = Satrec.twoline2rv(
        tle1,
        tle2,
    )

    jd, fr = jday(
        target_time.year,
        target_time.month,
        target_time.day,
        target_time.hour,
        target_time.minute,
        target_time.second
        + target_time.microsecond / 1_000_000,
    )

    error, position, velocity = satellite.sgp4(
        jd,
        fr,
    )

    if error != 0:
        raise RuntimeError(
            f"SGP4 failed with error code {error}"
        )

    return np.array(position)


# --------------------------------------------------
# Recompute errors
# --------------------------------------------------

results = []

for _, row in selected.iterrows():

    source = tle_lookup.loc[
        row["source_creation"]
    ]

    reference = tle_lookup.loc[
        row["reference_creation"]
    ]

    source_position = propagate(
        source["TLE_LINE1"],
        source["TLE_LINE2"],
        row["target_time"],
    )

    reference_position = propagate(
        reference["TLE_LINE1"],
        reference["TLE_LINE2"],
        row["target_time"],
    )

    recomputed_error = np.linalg.norm(
        source_position - reference_position
    )

    difference = abs(
        row["proxy_error_km"]
        - recomputed_error
    )

    results.append(
        {
            "horizon_hours": row["horizon_hours"],
            "stored_error_km": row["proxy_error_km"],
            "recomputed_error_km": recomputed_error,
            "difference_km": difference,
        }
    )


# --------------------------------------------------
# Report
# --------------------------------------------------

results = pd.DataFrame(results)

print()
print(
    "Horizon | Stored km | Recomputed km | Difference"
)
print("-" * 55)

for _, row in results.iterrows():

    print(
        f"{row['horizon_hours']:7.0f} | "
        f"{row['stored_error_km']:10.6f} | "
        f"{row['recomputed_error_km']:13.6f} | "
        f"{row['difference_km']:.9f}"
    )

print()
print(
    "Maximum difference:",
    results["difference_km"].max(),
    "km",
)