
import pandas as pd
from pathlib import Path


# ============================================================
# O-RAP Temporal Split Builder
# ============================================================

FEATURE_FILE = Path(
    "data/processed/orap_features_2025.csv"
)

LABEL_FILE = Path(
    "data/processed/orap_multisatellite_labels_2025.csv"
)

RAW_DIR = Path(
    "data/raw"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("O-RAP TEMPORAL SPLIT BUILDER")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load feature and label datasets
    # --------------------------------------------------------

    features = pd.read_csv(
        FEATURE_FILE
    )

    labels = pd.read_csv(
        LABEL_FILE
    )

    print(
        f"Feature samples: {len(features)}"
    )

    print(
        f"Original label samples: {len(labels)}"
    )

    # --------------------------------------------------------
    # 2. Parse label timestamps
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

    # --------------------------------------------------------
    # 3. Load raw TLE metadata
    # --------------------------------------------------------

    raw_frames = []

    for path in sorted(
        RAW_DIR.glob("*_2025.csv")
    ):

        if path.name == "multisatellite_2025_summary.csv":
            continue

        raw_frames.append(
            pd.read_csv(
                path,
                usecols=[
                    "NORAD_CAT_ID",
                    "CREATION_DATE",
                    "EPOCH",
                    "TLE_LINE1",
                    "TLE_LINE2",
                ],
            )
        )

    if not raw_frames:
        raise RuntimeError(
            "No raw satellite CSV files found."
        )

    raw = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    print(
        f"Raw TLE records: {len(raw)}"
    )

    # --------------------------------------------------------
    # 4. Parse raw timestamps
    # --------------------------------------------------------

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
    # 5. Remove exact duplicate TLE records
    # --------------------------------------------------------

    raw = raw.drop_duplicates(
        subset=[
            "NORAD_CAT_ID",
            "TLE_LINE1",
            "TLE_LINE2",
        ],
        keep="first",
    ).copy()

    # --------------------------------------------------------
    # 6. Identify ambiguous source events
    # --------------------------------------------------------
    #
    # Same definition used by build_features.py:
    #
    #   NORAD_CAT_ID
    #   CREATION_DATE
    #   EPOCH
    #
    # If multiple different TLE records share this complete
    # source-event key, we exclude the associated labels.
    # --------------------------------------------------------

    key_counts = (
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

    ambiguous_keys = key_counts[
        key_counts["match_count"] > 1
    ].copy()

    print(
        "Ambiguous source-event keys: "
        f"{len(ambiguous_keys)}"
    )

    # --------------------------------------------------------
    # 7. Remove the same six ambiguous labels
    # --------------------------------------------------------

    if len(ambiguous_keys) > 0:

        ambiguous_index = pd.MultiIndex.from_frame(
            ambiguous_keys[
                [
                    "NORAD_CAT_ID",
                    "CREATION_DATE",
                    "EPOCH",
                ]
            ].rename(
                columns={
                    "NORAD_CAT_ID": "norad_id",
                    "CREATION_DATE": "source_creation",
                    "EPOCH": "source_epoch",
                }
            )
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

    excluded = int(
        ambiguous_labels.sum()
    )

    print(
        "Ambiguous labels excluded: "
        f"{excluded}"
    )

    labels = labels.loc[
        ~ambiguous_labels
    ].copy()

    # --------------------------------------------------------
    # 8. Verify feature/label row counts
    # --------------------------------------------------------
    #
    # build_features.py used exactly this exclusion and then
    # constructed the feature rows from these labels.
    #
    # Therefore the resulting label count must equal the
    # feature count.
    # --------------------------------------------------------

    if len(labels) != len(features):

        raise RuntimeError(
            "Feature/label row-count mismatch.\n"
            f"Features: {len(features)}\n"
            f"Labels after exclusion: {len(labels)}"
        )

    print(
        "Feature/label rows aligned: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # 9. Verify row-level identity
    # --------------------------------------------------------
    #
    # We cannot use source_creation directly from the feature
    # file because it was intentionally excluded from the
    # modeling dataset.
    #
    # Instead, verify that the label and feature target columns
    # agree row-by-row.
    # --------------------------------------------------------

    feature_target = (
        features[
            "unreliable_1km"
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    label_target = (
        (
            labels[
                "proxy_error_km"
            ] > 1.0
        )
        .astype(int)
        .reset_index(drop=True)
    )

    target_matches = (
        feature_target == label_target
    )

    if not target_matches.all():

        mismatches = int(
            (~target_matches).sum()
        )

        raise RuntimeError(
            f"Feature/label target mismatch: "
            f"{mismatches} rows."
        )

    print(
        "Target alignment verified."
    )

    # --------------------------------------------------------
    # 10. Attach timestamps ONLY for split construction
    # --------------------------------------------------------
    #
    # These timestamps will NOT be included in the saved
    # modeling datasets.
    # --------------------------------------------------------

    split_index = pd.DataFrame({
        "source_creation":
            labels["source_creation"].reset_index(
                drop=True
            )
    })

    split_index["month"] = (
        split_index[
            "source_creation"
        ].dt.month
    )

    # --------------------------------------------------------
    # 11. Create temporal masks
    # --------------------------------------------------------

    train_mask = (
        split_index["month"].between(
            1,
            8,
        )
    )

    validation_mask = (
        split_index["month"].between(
            9,
            10,
        )
    )

    test_mask = (
        split_index["month"].between(
            11,
            12,
        )
    )

    # --------------------------------------------------------
    # 12. Verify every sample belongs to exactly one split
    # --------------------------------------------------------

    split_count = (
        train_mask.astype(int)
        + validation_mask.astype(int)
        + test_mask.astype(int)
    )

    if not (split_count == 1).all():

        raise RuntimeError(
            "Some samples were assigned to zero or "
            "multiple temporal splits."
        )

    # --------------------------------------------------------
    # 13. Apply masks to feature rows
    # --------------------------------------------------------

    train = features.loc[
        train_mask.values
    ].copy()

    validation = features.loc[
        validation_mask.values
    ].copy()

    test = features.loc[
        test_mask.values
    ].copy()

    # --------------------------------------------------------
    # 14. Reset indices
    # --------------------------------------------------------

    train.reset_index(
        drop=True,
        inplace=True,
    )

    validation.reset_index(
        drop=True,
        inplace=True,
    )

    test.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # 15. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = (
        OUTPUT_DIR
        / "train_temporal.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / "validation_temporal.csv"
    )

    test_file = (
        OUTPUT_DIR
        / "test_temporal.csv"
    )

    # --------------------------------------------------------
    # 16. Save
    # --------------------------------------------------------

    train.to_csv(
        train_file,
        index=False,
    )

    validation.to_csv(
        validation_file,
        index=False,
    )

    test.to_csv(
        test_file,
        index=False,
    )

    # --------------------------------------------------------
    # 17. Report split date ranges
    # --------------------------------------------------------

    train_dates = split_index.loc[
        train_mask,
        "source_creation",
    ]

    validation_dates = split_index.loc[
        validation_mask,
        "source_creation",
    ]

    test_dates = split_index.loc[
        test_mask,
        "source_creation",
    ]

    print()
    print("=" * 90)
    print("TEMPORAL SPLIT COMPLETE")
    print("=" * 90)

    print(
        f"Train:      {len(train):5d} samples "
        "(Jan-Aug)"
    )

    print(
        f"Validation: {len(validation):5d} samples "
        "(Sep-Oct)"
    )

    print(
        f"Test:       {len(test):5d} samples "
        "(Nov-Dec)"
    )

    print()

    print(
        "Train period:"
    )

    print(
        f"  {train_dates.min()} "
        f"-> {train_dates.max()}"
    )

    print()

    print(
        "Validation period:"
    )

    print(
        f"  {validation_dates.min()} "
        f"-> {validation_dates.max()}"
    )

    print()

    print(
        "Test period:"
    )

    print(
        f"  {test_dates.min()} "
        f"-> {test_dates.max()}"
    )

    # --------------------------------------------------------
    # 18. Class balance
    # --------------------------------------------------------

    print()
    print("Class balance:")

    for name, dataset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        counts = (
            dataset[
                "unreliable_1km"
            ]
            .value_counts()
            .sort_index()
        )

        total = len(dataset)

        positive = int(
            counts.get(1, 0)
        )

        percentage = (
            100.0 * positive / total
        )

        print(
            f"{name:10s}: "
            f"0={int(counts.get(0, 0)):5d}, "
            f"1={positive:5d}, "
            f"positive={percentage:.2f}%"
        )

    # --------------------------------------------------------
    # 19. Satellite distribution
    # --------------------------------------------------------

    print()
    print("Satellite distribution:")

    print(
        pd.DataFrame({
            "Train":
                train["satellite_name"]
                .value_counts(),

            "Validation":
                validation["satellite_name"]
                .value_counts(),

            "Test":
                test["satellite_name"]
                .value_counts(),
        })
        .fillna(0)
        .astype(int)
        .to_string()
    )

    # --------------------------------------------------------
    # 20. Horizon distribution
    # --------------------------------------------------------

    print()
    print("Horizon distribution:")

    print(
        pd.DataFrame({
            "Train":
                train["horizon_hours"]
                .value_counts(),

            "Validation":
                validation["horizon_hours"]
                .value_counts(),

            "Test":
                test["horizon_hours"]
                .value_counts(),
        })
        .fillna(0)
        .astype(int)
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 21. Final file report
    # --------------------------------------------------------

    print()
    print("Output files:")

    print(
        f"  {train_file}"
    )

    print(
        f"  {validation_file}"
    )

    print(
        f"  {test_file}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
