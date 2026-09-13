
USERNAME = "aimrrs404@gmail.com"
PASSWORD = "qwertyuioplkjhgfdsa"
import requests
import pandas as pd

# --------------------------------------------------
# Satellite cohort
# --------------------------------------------------

SATELLITES = {
    25544: "ISS",
    20580: "Hubble Space Telescope",
    33591: "NOAA 19",
    39634: "Sentinel-1A",
    24876: "TDRS-5",
    26407: "GPS BIIR-2",
}

# --------------------------------------------------
# Dataset window
# --------------------------------------------------

START = "2025-01-01"
END = "2025-04-01"

# --------------------------------------------------
# Credentials
# --------------------------------------------------


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
# Coverage check
# --------------------------------------------------

results = []

print()
print("=" * 100)
print(f"COVERAGE WINDOW: {START} -> {END}")
print("=" * 100)

for norad_id, name in SATELLITES.items():

    print(f"\nChecking {name} ({norad_id})...")

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"
        f"CREATION_DATE/{START}--{END}/"
        "orderby/CREATION_DATE%20asc/"
        "format/json"
    )

    response = session.get(url)

    print("  Query status:", response.status_code)

    if response.status_code != 200:
        print("  Request failed.")
        continue

    records = response.json()

    if not records:
        print("  No records found.")
        continue

    df = pd.DataFrame(records)

    # --------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------

    df["CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        utc=True,
    )

    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    df = df.sort_values(
        "CREATION_DATE"
    ).reset_index(drop=True)

    records_returned = len(df)

    # --------------------------------------------------
    # Remove exact duplicate TLEs
    # --------------------------------------------------

    unique_df = df.drop_duplicates(
        subset=["TLE_LINE1", "TLE_LINE2"],
        keep="first",
    ).reset_index(drop=True)

    unique_tles = len(unique_df)

    # --------------------------------------------------
    # Calculate update intervals
    # --------------------------------------------------

    intervals = (
        unique_df["CREATION_DATE"]
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

    max_interval = (
        intervals.max()
        if not intervals.empty
        else None
    )

    # --------------------------------------------------
    # Print satellite result
    # --------------------------------------------------

    print("  Records returned:", records_returned)
    print("  Unique TLEs:", unique_tles)
    print("  First creation:", unique_df["CREATION_DATE"].iloc[0])
    print("  Last creation:", unique_df["CREATION_DATE"].iloc[-1])

    if median_interval is not None:
        print(
            f"  Median update interval: "
            f"{median_interval:.2f} hours"
        )

        print(
            f"  Maximum update gap: "
            f"{max_interval:.2f} hours"
        )

    # --------------------------------------------------
    # Store result
    # --------------------------------------------------

    results.append(
        {
            "norad_id": norad_id,
            "name": name,
            "records_returned": records_returned,
            "unique_tles": unique_tles,
            "first_creation": unique_df[
                "CREATION_DATE"
            ].iloc[0],
            "last_creation": unique_df[
                "CREATION_DATE"
            ].iloc[-1],
            "median_interval_hours": median_interval,
            "max_gap_hours": max_interval,
        }
    )

# --------------------------------------------------
# Final summary
# --------------------------------------------------

summary = pd.DataFrame(results)

print()
print("=" * 100)
print("MULTI-SATELLITE COVERAGE SUMMARY")
print("=" * 100)

if summary.empty:

    print("No satellite data returned.")

else:

    print(
        summary.to_string(
            index=False
        )
    )

print()
print("=" * 100)
print("Coverage check complete")
print("=" * 100)