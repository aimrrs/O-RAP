import pandas as pd
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import timedelta

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

HORIZONS_HOURS = [6, 12, 24, 48]

MAX_REFERENCE_DELAY_HOURS = 6

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR / "orap_multisatellite_labels_2025.csv"
)

# --------------------------------------------------
# SGP4 propagation helper
# --------------------------------------------------

def propagate(tle1, tle2, target_time):

    satellite = Satrec.twoline2rv(
        tle1,
        tle2,
    )

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

    error, position, velocity = satellite.sgp4(
        jd,
        fr,
    )

    if error != 0:
        return None

    return np.array(position)


# --------------------------------------------------
# Load satellite data
# --------------------------------------------------

def load_satellite(norad_id, name):

    filename = (
        f"{norad_id}_"
        f"{name.replace(' ', '_')}_2025.csv"
    )

    path = RAW_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Missing dataset: {path}"
        )

    df = pd.read_csv(path)

    df["CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        utc=True,
    )

    df["EPOCH"] = pd.to_datetime(
        df["EPOCH"],
        format="mixed",
        utc=True,
    )

    # Ensure chronological order.
    df = df.sort_values(
        "CREATION_DATE"
    ).reset_index(drop=True)

    # Defensive duplicate removal.
    df = df.drop_duplicates(
        subset=[
            "TLE_LINE1",
            "TLE_LINE2",
        ],
        keep="first",
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# Build labels for one satellite
# --------------------------------------------------

def build_labels(norad_id, name):

    print()
    print("=" * 90)
    print(
        f"Processing {name} ({norad_id})"
    )
    print("=" * 90)

    df = load_satellite(
        norad_id,
        name,
    )

    print(
        "Unique TLEs:",
        len(df),
    )

    rows = []

    # --------------------------------------------------
    # Iterate through available source TLEs
    # --------------------------------------------------

    for i, source in df.iterrows():

        source_creation = (
            source["CREATION_DATE"]
        )

        # --------------------------------------------------
        # Calculate TLE age at source availability
        # --------------------------------------------------

        tle_age_hours = (
            source_creation
            - source["EPOCH"]
        ).total_seconds() / 3600

        # --------------------------------------------------
        # Test each prediction horizon
        # --------------------------------------------------

        for horizon in HORIZONS_HOURS:

            target_time = (
                source_creation
                + timedelta(hours=horizon)
            )

            # --------------------------------------------------
            # Find first TLE available at/after target
            # --------------------------------------------------

            candidates = df[
                df["CREATION_DATE"]
                >= target_time
            ]

            if candidates.empty:
                continue

            reference = candidates.iloc[0]

            reference_creation = (
                reference["CREATION_DATE"]
            )

            reference_delay_hours = (
                reference_creation
                - target_time
            ).total_seconds() / 3600

            # --------------------------------------------------
            # Enforce maximum reference delay
            # --------------------------------------------------

            if (
                reference_delay_hours
                > MAX_REFERENCE_DELAY_HOURS
            ):
                continue

            # --------------------------------------------------
            # Propagate source TLE
            # --------------------------------------------------

            predicted_position = propagate(
                source["TLE_LINE1"],
                source["TLE_LINE2"],
                target_time,
            )

            if predicted_position is None:
                continue

            # --------------------------------------------------
            # Propagate later reference TLE
            # --------------------------------------------------

            reference_position = propagate(
                reference["TLE_LINE1"],
                reference["TLE_LINE2"],
                target_time,
            )

            if reference_position is None:
                continue

            # --------------------------------------------------
            # Calculate 3D position divergence
            # --------------------------------------------------

            error_km = np.linalg.norm(
                predicted_position
                - reference_position
            )

            # --------------------------------------------------
            # Save sample
            # --------------------------------------------------

            rows.append(
                {
                    "norad_id": norad_id,
                    "satellite_name": name,

                    "source_creation":
                        source_creation,

                    "target_time":
                        target_time,

                    "horizon_hours":
                        horizon,

                    "reference_creation":
                        reference_creation,

                    "source_epoch":
                        source["EPOCH"],

                    "reference_epoch":
                        reference["EPOCH"],

                    "source_tle_age_hours":
                        tle_age_hours,

                    "reference_delay_hours":
                        reference_delay_hours,

                    "proxy_error_km":
                        error_km,
                }
            )

    result = pd.DataFrame(rows)

    print(
        "Usable labeled samples:",
        len(result),
    )

    if not result.empty:

        print()
        print("Samples by horizon:")

        print(
            result[
                "horizon_hours"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print("Proxy error statistics:")

        print(
            result[
                "proxy_error_km"
            ].describe()
        )

    return result


# --------------------------------------------------
# Main processing
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

all_results = []

for norad_id, name in SATELLITES.items():

    result = build_labels(
        norad_id,
        name,
    )

    if not result.empty:
        all_results.append(result)


# --------------------------------------------------
# Combine satellites
# --------------------------------------------------

if not all_results:

    raise RuntimeError(
        "No labeled samples were generated."
    )

combined = pd.concat(
    all_results,
    ignore_index=True,
)

# --------------------------------------------------
# Final chronological ordering
# --------------------------------------------------

combined = combined.sort_values(
    [
        "satellite_name",
        "target_time",
        "horizon_hours",
    ]
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)


# --------------------------------------------------
# Final report
# --------------------------------------------------

print()
print("=" * 100)
print("O-RAP MULTI-SATELLITE LABEL GENERATION COMPLETE")
print("=" * 100)

print(
    "Total labeled samples:",
    len(combined),
)

print(
    "Satellites:",
    combined[
        "satellite_name"
    ].nunique(),
)

print()
print("Samples by satellite:")

print(
    combined[
        "satellite_name"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Samples by horizon:")

print(
    combined[
        "horizon_hours"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Proxy error statistics:")

print(
    combined[
        "proxy_error_km"
    ].describe()
)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("=" * 100)