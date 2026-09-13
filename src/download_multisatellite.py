import requests
import pandas as pd
from pathlib import Path
import time

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SATELLITES = {
    25544: "ISS",
    20580: "Hubble Space Telescope",
    33591: "NOAA 19",
    39634: "Sentinel-1A",
    24876: "TDRS-5",
    26407: "GPS BIIR-2",
}

START = "2025-01-01"
END = "2026-01-01"

OUTPUT_DIR = Path("data/raw")

LOGIN_URL = (
    "https://www.space-track.org/ajaxauth/login"
)

# --------------------------------------------------
# Credentials
# --------------------------------------------------

USERNAME = "aimrrs404@gmail.com"
PASSWORD = "qwertyuioplkjhgfdsa"

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
    raise RuntimeError(
        "Space-Track login failed."
    )

# --------------------------------------------------
# Prepare output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------
# Download each satellite
# --------------------------------------------------

summary = []

for norad_id, name in SATELLITES.items():

    print()
    print("=" * 90)
    print(f"Downloading: {name} ({norad_id})")
    print("=" * 90)

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"
        f"CREATION_DATE/{START}--{END}/"
        "orderby/CREATION_DATE%20asc/"
        "format/json"
    )

    response = session.get(url)

    print(
        "Query status:",
        response.status_code
    )

    if response.status_code != 200:
        print(
            "Download failed for",
            name
        )
        continue

    records = response.json()

    if not records:
        print("No records returned.")
        continue

    df = pd.DataFrame(records)

    # --------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------

    df["CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        utc=True,
    )

    df["EPOCH"] = pd.to_datetime(
        df["EPOCH"],
        utc=True,
    )

    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    df = df.sort_values(
        "CREATION_DATE"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Remove exact duplicate TLE pairs
    # --------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "TLE_LINE1",
            "TLE_LINE2",
        ],
        keep="first",
    ).reset_index(drop=True)

    duplicates_removed = (
        before - len(df)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{norad_id}_{name.replace(' ', '_')}_2025.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    intervals = (
        df["CREATION_DATE"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 3600
    )

    median_interval = (
        intervals.median()
        if not intervals.empty
        else None
    )

    max_gap = (
        intervals.max()
        if not intervals.empty
        else None
    )

    print(
        "Records downloaded:",
        before
    )

    print(
        "Exact duplicates removed:",
        duplicates_removed
    )

    print(
        "Unique TLEs:",
        len(df)
    )

    print(
        "First creation:",
        df["CREATION_DATE"].iloc[0]
    )

    print(
        "Last creation:",
        df["CREATION_DATE"].iloc[-1]
    )

    if median_interval is not None:
        print(
            f"Median update interval: "
            f"{median_interval:.2f} hours"
        )

        print(
            f"Maximum update gap: "
            f"{max_gap:.2f} hours"
        )

    print(
        "Saved:",
        output_file
    )

    summary.append(
        {
            "norad_id": norad_id,
            "name": name,
            "records_downloaded": before,
            "duplicates_removed": duplicates_removed,
            "unique_tles": len(df),
            "median_interval_hours": median_interval,
            "max_gap_hours": max_gap,
        }
    )

    # Be polite to Space-Track.
    time.sleep(2)

# --------------------------------------------------
# Save acquisition summary
# --------------------------------------------------

if summary:

    summary_df = pd.DataFrame(summary)

    summary_file = (
        OUTPUT_DIR
        / "multisatellite_2025_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 90)
    print("MULTI-SATELLITE DATASET ACQUISITION COMPLETE")
    print("=" * 90)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary saved:",
        summary_file
    )

else:

    print()
    print(
        "No satellite datasets were downloaded."
    )