import os
import requests
import pandas as pd


# --------------------------------------------------
# Candidate satellites
# --------------------------------------------------

SATELLITES = {
    25544: "ISS",
    20580: "Hubble Space Telescope",
    33591: "NOAA 19",
    25994: "Terra",
    27424: "Aqua",
    39634: "Sentinel-1A",
    24876: "TDRS-5",
    26407: "GPS BIIR-2",
}


USERNAME = os.environ.get("SPACE_TRACK_IDENTITY")
PASSWORD = os.environ.get("SPACE_TRACK_PASSWORD")
if not USERNAME or not PASSWORD:
    raise RuntimeError("Set SPACE_TRACK_IDENTITY and SPACE_TRACK_PASSWORD before using Space-Track.")


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
# Check historical coverage
# --------------------------------------------------

results = []

for norad_id, name in SATELLITES.items():

    print(f"\nChecking {name} ({norad_id})...")

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"
        "CREATION_DATE/%3E2024-01-01/"
        "orderby/CREATION_DATE%20asc/"
        "limit/1000/"
        "format/json"
    )

    response = session.get(url)

    if response.status_code != 200:
        print("  Request failed:", response.status_code)
        continue

    records = response.json()

    if not records:
        print("  No records found.")
        continue

    df = pd.DataFrame(records)

    df["CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        utc=True,
    )

    results.append(
        {
            "norad_id": norad_id,
            "name": name,
            "records_returned": len(df),
            "first_creation": df["CREATION_DATE"].min(),
            "last_creation": df["CREATION_DATE"].max(),
        }
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = pd.DataFrame(results)

print()
print("=" * 80)
print("SATELLITE COVERAGE SUMMARY")
print("=" * 80)

if summary.empty:
    print("No satellite data returned.")
else:
    print(
        summary.to_string(index=False)
    )

    print()
    print(
        "Note: records_returned is capped at 1000."
    )
