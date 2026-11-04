
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ============================================================
# O-RAP: SOURCE-TLE FEATURE EXTRACTION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "results" / "modeling" / "splits"
OUTPUT_DIR = ROOT / "results" / "modeling" / "features"

PARTITIONS = ["train", "validation", "test"]

# Exactly 13 source-TLE orbital features.
ORBITAL_FIELDS = {
    "mean_motion": "MEAN_MOTION",
    "eccentricity": "ECCENTRICITY",
    "inclination": "INCLINATION",
    "raan": "RA_OF_ASC_NODE",
    "argument_of_pericenter": "ARG_OF_PERICENTER",
    "mean_anomaly": "MEAN_ANOMALY",
    "bstar": "BSTAR",
    "mean_motion_dot": "MEAN_MOTION_DOT",
    "mean_motion_ddot": "MEAN_MOTION_DDOT",
    "semimajor_axis": "SEMIMAJOR_AXIS",
    "orbital_period": "PERIOD",
    "apoapsis": "APOAPSIS",
    "periapsis": "PERIAPSIS",
}

FEATURE_COLUMNS = ["interval_hours"] + list(ORBITAL_FIELDS.keys())

METADATA_COLUMNS = [
    "pair_id",
    "satellite_key",
    "satellite_name",
    "norad_id",
    "source_gp_id",
    "reference_gp_id",
    "source_epoch",
    "reference_epoch",
]

TARGET_COLUMN = "label_1km"


def parse_utc(value):
    """Parse timestamps and normalize them to UTC."""

    dt = datetime.fromisoformat(value.strip())

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def load_raw_records(filename, cache):
    """Load and cache one original satellite JSON file."""

    if filename not in cache:

        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Raw JSON file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list in {filename}"
            )

        cache[filename] = records

    return cache[filename]


def extract_source_features(row, cache):
    """Extract verified features from the source TLE record."""

    filename = row["raw_file"]
    record_index = int(row["source_record_index"])

    records = load_raw_records(filename, cache)

    if record_index < 0 or record_index >= len(records):
        raise IndexError(
            f"Invalid source index {record_index} in {filename}"
        )

    record = records[record_index]

    # --------------------------------------------------------
    # Verify original record identity
    # --------------------------------------------------------

    if str(record["GP_ID"]) != str(row["source_gp_id"]):
        raise ValueError(
            f"GP_ID mismatch for pair {row['pair_id']}: "
            f"CSV={row['source_gp_id']}, "
            f"JSON={record['GP_ID']}"
        )

    if str(record["NORAD_CAT_ID"]) != str(row["norad_id"]):
        raise ValueError(
            f"NORAD mismatch for pair {row['pair_id']}"
        )

    json_epoch = parse_utc(record["EPOCH"])
    csv_epoch = parse_utc(row["source_epoch"])

    epoch_difference = abs(
        (json_epoch - csv_epoch).total_seconds()
    )

    if epoch_difference > 0.002:
        raise ValueError(
            f"Source epoch mismatch for pair {row['pair_id']}: "
            f"{epoch_difference} seconds"
        )

    # --------------------------------------------------------
    # Extract numeric source orbital features
    # --------------------------------------------------------

    features = {}

    interval = float(row["interval_hours"])

    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"Invalid interval for pair {row['pair_id']}"
        )

    features["interval_hours"] = interval

    for feature_name, json_field in ORBITAL_FIELDS.items():

        value = float(record[json_field])

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite {json_field} in pair {row['pair_id']}"
            )

        features[feature_name] = value

    # Validate target.
    label = int(row[TARGET_COLUMN])

    if label not in (0, 1):
        raise ValueError(
            f"Invalid label in pair {row['pair_id']}"
        )

    return features, label


def main():

    print("=" * 72)
    print("O-RAP SOURCE-TLE FEATURE EXTRACTION")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_cache = {}

    audit = {
        "total_rows": 0,
        "successful_rows": 0,
        "failed_rows": 0,
        "positive_labels": 0,
        "negative_labels": 0,
    }

    satellite_counts = defaultdict(Counter)
    errors = []

    output_fields = (
        METADATA_COLUMNS
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    for partition in PARTITIONS:

        input_file = SPLIT_DIR / f"{partition}.csv"

        if not input_file.exists():
            raise FileNotFoundError(input_file)

        output_file = (
            OUTPUT_DIR / f"{partition}_features.csv"
        )

        print(f"\nProcessing {partition.upper()}...")

        with input_file.open(
            "r", encoding="utf-8-sig", newline=""
        ) as source, output_file.open(
            "w", encoding="utf-8", newline=""
        ) as destination:

            reader = csv.DictReader(source)

            writer = csv.DictWriter(
                destination,
                fieldnames=output_fields
            )

            writer.writeheader()

            for row in reader:

                audit["total_rows"] += 1

                try:
                    features, label = extract_source_features(
                        row, raw_cache
                    )

                    output_row = {}

                    for column in METADATA_COLUMNS:
                        output_row[column] = row[column]

                    output_row.update(features)
                    output_row[TARGET_COLUMN] = label

                    writer.writerow(output_row)

                    audit["successful_rows"] += 1

                    if label == 1:
                        audit["positive_labels"] += 1
                    else:
                        audit["negative_labels"] += 1

                    satellite_counts[
                        (partition, row["satellite_name"])
                    ]["rows"] += 1

                    satellite_counts[
                        (partition, row["satellite_name"])
                    ][f"label_{label}"] += 1

                except Exception as error:

                    audit["failed_rows"] += 1

                    errors.append({
                        "partition": partition,
                        "pair_id": row.get("pair_id", ""),
                        "error": str(error),
                    })

                    print(
                        f"ERROR: pair {row.get('pair_id')}: {error}"
                    )

        print(
            f"{partition.upper()} output: {output_file}"
        )

    # --------------------------------------------------------
    # Write extraction audit
    # --------------------------------------------------------

    audit_file = OUTPUT_DIR / "feature_extraction_audit.txt"

    with audit_file.open("w", encoding="utf-8") as file:

        file.write("O-RAP FEATURE EXTRACTION AUDIT\n")
        file.write("=" * 65 + "\n\n")

        for key, value in audit.items():
            file.write(f"{key}: {value:,}\n")

        file.write("\nFEATURES\n")
        file.write("-" * 65 + "\n")

        for feature in FEATURE_COLUMNS:
            file.write(feature + "\n")

        file.write("\nSATELLITE DISTRIBUTION\n")
        file.write("-" * 65 + "\n")

        for (partition, satellite), counts in sorted(
            satellite_counts.items()
        ):

            file.write(
                f"{partition},{satellite},"
                f"rows={counts['rows']},"
                f"positive={counts['label_1']},"
                f"negative={counts['label_0']}\n"
            )

        if errors:
            file.write("\nERRORS\n")
            file.write("-" * 65 + "\n")

            for error in errors:
                file.write(
                    f"{error['partition']},"
                    f"{error['pair_id']},"
                    f"{error['error']}\n"
                )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 72)

    print(f"Total rows:       {audit['total_rows']:,}")
    print(f"Successful rows:  {audit['successful_rows']:,}")
    print(f"Failed rows:      {audit['failed_rows']:,}")
    print(f"Positive labels:  {audit['positive_labels']:,}")
    print(f"Negative labels:  {audit['negative_labels']:,}")

    print(f"\nFeatures extracted: {len(FEATURE_COLUMNS)}")

    for index, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{index:2d}. {feature}")

    if audit["failed_rows"] != 0:
        raise RuntimeError(
            "Feature extraction failed. Inspect the audit before modelling."
        )

    if audit["successful_rows"] != audit["total_rows"]:
        raise RuntimeError(
            "Input and output row counts do not match."
        )

    print("\n[PASS] All source records verified.")
    print("[PASS] All feature values finite.")
    print("[PASS] No extraction failures.")
    print("[PASS] All labels preserved.")
    print("\nSUCCESS: O-RAP feature datasets created.")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
