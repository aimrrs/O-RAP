
import pandas as pd
from pathlib import Path


# ============================================================
# O-RAP Feature Dataset Builder
# ============================================================

RAW_DIR = Path("data/raw")
LABEL_FILE = Path("data/processed/orap_multisatellite_labels_2025.csv")
OUTPUT_FILE = Path("data/processed/orap_features_2025.csv")


# ============================================================
# Leakage-safe modeling columns
# ============================================================
#
# These are available from the source TLE and prediction setup.
#
# Deliberately excluded:
#   - proxy_error_km
#   - reference_creation
#   - reference_epoch
#   - reference_delay_hours
#   - target_time
#
# Those contain information that is only known after the
# prediction target time and therefore must not enter the model.
# ============================================================

FEATURE_COLUMNS = [
    "norad_id",
    "satellite_name",
    "horizon_hours",
    "source_tle_age_hours",
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "period",
    "apoapsis",
    "periapsis",
    "unreliable_1km",
]


def main():

    print("=" * 90)
    print("O-RAP FEATURE DATASET BUILDER")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load label dataset
    # --------------------------------------------------------

    labels = pd.read_csv(LABEL_FILE)

    print(f"Loaded labels: {len(labels)}")

    # --------------------------------------------------------
    # 2. Load raw satellite datasets
    # --------------------------------------------------------

    raw_frames = []

    for path in sorted(RAW_DIR.glob("*_2025.csv")):

        # Do not load the summary file as a satellite dataset.
        if path.name == "multisatellite_2025_summary.csv":
            continue

        print(f"Loading: {path.name}")

        df = pd.read_csv(path)

        required_columns = [
            "NORAD_CAT_ID",
            "OBJECT_NAME",
            "CREATION_DATE",
            "EPOCH",
            "MEAN_MOTION",
            "ECCENTRICITY",
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "BSTAR",
            "MEAN_MOTION_DOT",
            "MEAN_MOTION_DDOT",
            "SEMIMAJOR_AXIS",
            "PERIOD",
            "APOAPSIS",
            "PERIAPSIS",
            "TLE_LINE1",
            "TLE_LINE2",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise RuntimeError(
                f"{path.name} is missing columns: "
                f"{missing_columns}"
            )

        raw_frames.append(
            df[required_columns].copy()
        )

    if not raw_frames:
        raise RuntimeError(
            "No raw satellite CSV files were found."
        )

    raw = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    print(
        f"Raw records loaded: {len(raw)}"
    )

    # --------------------------------------------------------
    # 3. Parse timestamps
    # --------------------------------------------------------

    labels["source_creation"] = pd.to_datetime(
        labels["source_creation"],
        format="mixed",
        utc=True,
    )

    labels["source_epoch"] = pd.to_datetime(
        labels["source_epoch"],
        format="mixed",
        utc=True,
    )

    raw["CREATION_DATE"] = pd.to_datetime(
        raw["CREATION_DATE"],
        format="mixed",
        utc=True,
    )

    raw["EPOCH"] = pd.to_datetime(
        raw["EPOCH"],
        format="mixed",
        utc=True,
    )

    # --------------------------------------------------------
    # 4. Remove exact duplicate TLE records
    # --------------------------------------------------------
    #
    # Same policy used during label generation.
    #
    # Only records with identical complete TLE lines are removed.
    # Different TLEs are preserved even if timestamps match.
    # --------------------------------------------------------

    before_dedup = len(raw)

    raw = raw.drop_duplicates(
        subset=[
            "NORAD_CAT_ID",
            "TLE_LINE1",
            "TLE_LINE2",
        ],
        keep="first",
    ).copy()

    after_dedup = len(raw)

    print(
        "Exact duplicate TLE records removed: "
        f"{before_dedup - after_dedup}"
    )

    print(
        f"Unique raw TLE records: {after_dedup}"
    )

    # --------------------------------------------------------
    # 5. Identify ambiguous source-event keys
    # --------------------------------------------------------
    #
    # Normally a source event is identified by:
    #
    #   NORAD_CAT_ID + CREATION_DATE + EPOCH
    #
    # However, Space-Track contains a small number of cases where
    # different TLEs share all three fields.
    #
    # We do not arbitrarily choose between them.
    #
    # Instead, all labels associated with those ambiguous source
    # events are excluded.
    #
    # This is intentionally conservative and affects only a tiny
    # fraction of the dataset.
    # --------------------------------------------------------

    raw_key_counts = (
        raw.groupby(
            [
                "NORAD_CAT_ID",
                "CREATION_DATE",
                "EPOCH",
            ]
        )
        .size()
        .reset_index(
            name="match_count"
        )
    )

    ambiguous_keys = raw_key_counts[
        raw_key_counts["match_count"] > 1
    ].copy()

    print(
        "Ambiguous source-event keys found: "
        f"{len(ambiguous_keys)}"
    )

    # --------------------------------------------------------
    # 6. Convert ambiguous keys to label-key format
    # --------------------------------------------------------

    if len(ambiguous_keys) > 0:

        ambiguous_label_keys = (
            ambiguous_keys[
                [
                    "NORAD_CAT_ID",
                    "CREATION_DATE",
                    "EPOCH",
                ]
            ]
            .rename(
                columns={
                    "NORAD_CAT_ID": "norad_id",
                    "CREATION_DATE": "source_creation",
                    "EPOCH": "source_epoch",
                }
            )
        )

        ambiguous_index = pd.MultiIndex.from_frame(
            ambiguous_label_keys[
                [
                    "norad_id",
                    "source_creation",
                    "source_epoch",
                ]
            ]
        )

    else:

        ambiguous_index = pd.MultiIndex.from_tuples(
            [],
            names=[
                "norad_id",
                "source_creation",
                "source_epoch",
            ],
        )

    # --------------------------------------------------------
    # 7. Exclude ambiguous labels
    # --------------------------------------------------------

    label_index = pd.MultiIndex.from_frame(
        labels[
            [
                "norad_id",
                "source_creation",
                "source_epoch",
            ]
        ]
    )

    ambiguous_labels = label_index.isin(
        ambiguous_index
    )

    excluded_labels = int(
        ambiguous_labels.sum()
    )

    print(
        "Ambiguous labeled samples excluded: "
        f"{excluded_labels}"
    )

    labels = labels.loc[
        ~ambiguous_labels
    ].copy()

    # --------------------------------------------------------
    # 8. Exclude ambiguous raw TLE records
    # --------------------------------------------------------
    #
    # This guarantees that the remaining source-event key maps
    # to exactly one raw TLE.
    # --------------------------------------------------------

    raw_index = pd.MultiIndex.from_frame(
        raw[
            [
                "NORAD_CAT_ID",
                "CREATION_DATE",
                "EPOCH",
            ]
        ]
    )

    raw_ambiguous = raw_index.isin(
        pd.MultiIndex.from_frame(
            ambiguous_keys[
                [
                    "NORAD_CAT_ID",
                    "CREATION_DATE",
                    "EPOCH",
                ]
            ]
        )
    )

    raw = raw.loc[
        ~raw_ambiguous
    ].copy()

    print(
        f"Remaining labels: {len(labels)}"
    )

    print(
        f"Remaining raw TLE records: {len(raw)}"
    )

    # --------------------------------------------------------
    # 9. Merge labels with source-TLE orbital features
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # One source TLE can legitimately produce multiple labels:
    #
    #   same TLE -> 6h label
    #           -> 12h label
    #           -> 24h label
    #           -> 48h label
    #
    # Therefore the relationship is MANY-TO-ONE.
    # --------------------------------------------------------

    merged = labels.merge(
        raw,
        left_on=[
            "norad_id",
            "source_creation",
            "source_epoch",
        ],
        right_on=[
            "NORAD_CAT_ID",
            "CREATION_DATE",
            "EPOCH",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 10. Verify source-TLE matching
    # --------------------------------------------------------

    missing = int(
        merged["TLE_LINE1"].isna().sum()
    )

    if missing:
        raise RuntimeError(
            f"Could not match {missing} labeled samples "
            "to their source TLE."
        )

    print(
        "Successfully matched source TLEs: "
        f"{len(merged)}"
    )

    # --------------------------------------------------------
    # 11. Construct modeling dataset
    # --------------------------------------------------------
    #
    # Primary target:
    #
    #   unreliable_1km = 1
    #       if proxy_error_km > 1 km
    #
    #   unreliable_1km = 0
    #       otherwise
    #
    # proxy_error_km itself is NOT retained as a feature.
    # --------------------------------------------------------

    features = pd.DataFrame({

        "norad_id":
            merged["norad_id"],

        "satellite_name":
            merged["satellite_name"],

        "horizon_hours":
            merged["horizon_hours"],

        "source_tle_age_hours":
            merged["source_tle_age_hours"],

        "mean_motion":
            merged["MEAN_MOTION"],

        "eccentricity":
            merged["ECCENTRICITY"],

        "inclination":
            merged["INCLINATION"],

        "ra_of_asc_node":
            merged["RA_OF_ASC_NODE"],

        "arg_of_pericenter":
            merged["ARG_OF_PERICENTER"],

        "mean_anomaly":
            merged["MEAN_ANOMALY"],

        "bstar":
            merged["BSTAR"],

        "mean_motion_dot":
            merged["MEAN_MOTION_DOT"],

        "mean_motion_ddot":
            merged["MEAN_MOTION_DDOT"],

        "semimajor_axis":
            merged["SEMIMAJOR_AXIS"],

        "period":
            merged["PERIOD"],

        "apoapsis":
            merged["APOAPSIS"],

        "periapsis":
            merged["PERIAPSIS"],

        "unreliable_1km":
            (
                merged["proxy_error_km"] > 1.0
            ).astype(int),
    })

    # --------------------------------------------------------
    # 12. Enforce final column order
    # --------------------------------------------------------

    features = features[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # 13. Final integrity checks
    # --------------------------------------------------------

    missing_values = features.isna().sum()

    if missing_values.any():

        print(
            "Missing values detected:"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise RuntimeError(
            "Feature dataset contains missing values."
        )

    if len(features) == 0:
        raise RuntimeError(
            "Feature dataset is empty."
        )

    # Target must be binary.
    target_values = set(
        features[
            "unreliable_1km"
        ].unique()
    )

    if not target_values.issubset({0, 1}):

        raise RuntimeError(
            "Invalid target values: "
            f"{target_values}"
        )

    # --------------------------------------------------------
    # 14. Save feature dataset
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 15. Final report
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("O-RAP FEATURE DATASET COMPLETE")
    print("=" * 90)

    print(
        f"Final samples: {len(features)}"
    )

    print(
        f"Excluded ambiguous samples: "
        f"{excluded_labels}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("Class counts:")

    print(
        features[
            "unreliable_1km"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Class percentages:")

    class_percentages = (
        features[
            "unreliable_1km"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        class_percentages.round(2).to_string()
    )

    print()
    print("Samples by satellite:")

    print(
        features.groupby(
            "satellite_name"
        ).size().to_string()
    )

    print()
    print("Samples by horizon:")

    print(
        features.groupby(
            "horizon_hours"
        ).size().to_string()
    )

    print()
    print("Feature columns:")

    print(
        features.columns.tolist()
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
