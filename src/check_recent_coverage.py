import os
import requests
import pandas as pd

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

print()
print("=" * 100)
print("MOST RECENT TLE COVERAGE")
print("=" * 100)

for norad_id, name in SATELLITES.items():

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"
        "orderby/CREATION_DATE%20desc/"
        "limit/10/"
        "format/json"
    )

    response = session.get(url)

    if response.status_code != 200:
        print(f"\n{name} ({norad_id}) -> FAILED: {response.status_code}")
        continue

    records = response.json()

    if not records:
        print(f"\n{name} ({norad_id}) -> NO DATA")
        continue

    df = pd.DataFrame(records)

    df["CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        utc=True,
    )

    df = df.sort_values("CREATION_DATE")

    print(f"\n{name} ({norad_id})")
    print("-" * 70)
    print(
        "Oldest of recent 10:",
        df["CREATION_DATE"].iloc[0],
    )
    print(
        "Newest:",
        df["CREATION_DATE"].iloc[-1],
    )

    if len(df) >= 2:
        intervals = (
            df["CREATION_DATE"]
            .diff()
            .dropna()
            .dt.total_seconds()
            / 3600
        )

        print(
            "Recent update interval (hours): "
            f"median={intervals.median():.2f}, "
            f"min={intervals.min():.2f}, "
            f"max={intervals.max():.2f}"
        )

print()
print("=" * 100)
print("Coverage check complete")
print("=" * 100)
