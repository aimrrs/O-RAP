
"""
O-RAP: Batch Raw Dataset Audit

Audits every GP_HISTORY JSON file in data/raw.

Checks:
1. JSON readability and record counts
2. NORAD_CAT_ID consistency
3. Required-field completeness
4. EPOCH and CREATION_DATE validity and coverage
5. Duplicate GP_IDs
6. Duplicate TLE pairs
7. Repeated EPOCH groups and conflicting TLE pairs
8. TLE line lengths, prefixes and checksums
9. Yearly record distribution

Original raw files are never modified.

Outputs:
    results/data_audit/dataset_summary.csv
    results/data_audit/yearly_coverage.csv
    results/data_audit/data_quality_report.txt
"""

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "results" / "data_audit"

START_DATE = datetime(2005, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)

REQUIRED_FIELDS = [
    "NORAD_CAT_ID",
    "EPOCH",
    "CREATION_DATE",
    "GP_ID",
    "TLE_LINE1",
    "TLE_LINE2",
]

SATELLITE_NAMES = {
    "25544": "ISS",
    "25994": "Terra",
    "27424": "Aqua",
    "28376": "Aura",
    "28054": "DMSP 5D-3 F16",
    "29499": "METOP-A",
    "31598": "COSMO-SkyMed 1",
    "31698": "TerraSAR-X",
    "32382": "RADARSAT-2",
    "33591": "NOAA-19",
    "37849": "Suomi-NPP",
    "39084": "Landsat-8",
    "40697": "Sentinel-2A",
    "43013": "NOAA-20",
}


# ============================================================
# UTILITIES
# ============================================================

def parse_timestamp(value):
    """Parse an ISO timestamp and normalize it to UTC."""

    if not isinstance(value, str):
        raise ValueError("Timestamp is not a string")

    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def tle_checksum_valid(line):
    """
    Validate the standard TLE checksum.

    Sum all digits in columns 1-68.
    Each minus sign contributes 1.
    Take modulo 10 and compare with column 69.
    """

    if not isinstance(line, str) or len(line) != 69:
        return False

    if not line[68].isdigit():
        return False

    checksum = sum(
        int(char) if char.isdigit()
        else 1 if char == "-"
        else 0
        for char in line[:68]
    ) % 10

    return checksum == int(line[68])


def empty_summary(filename):
    return {
        "filename": filename,
        "satellite": "Unknown",
        "norad_id": "",
        "file_status": "NOT_PROCESSED",
        "records": 0,
        "earliest_epoch": "",
        "latest_epoch": "",
        "valid_epoch_records": 0,
        "out_of_range_epochs": 0,
        "invalid_epoch_records": 0,
        "invalid_creation_dates": 0,
        "missing_field_records": 0,
        "duplicate_gp_id_count": 0,
        "duplicate_tle_pair_count": 0,
        "repeated_epoch_groups": 0,
        "conflicting_epoch_groups": 0,
        "invalid_tle_line1": 0,
        "invalid_tle_line2": 0,
        "bad_tle_line1_checksums": 0,
        "bad_tle_line2_checksums": 0,
        "norad_mismatches": 0,
    }


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_file(file_path):

    filename = file_path.name
    result = empty_summary(filename)
    issues = []

    match = re.search(r"_(\d+)_gp_history\.json$", filename)

    if not match:
        result["file_status"] = "INVALID_FILENAME"
        return result, [], [
            f"{filename}: Could not determine NORAD ID from filename."
        ]

    expected_norad = match.group(1)

    result["norad_id"] = expected_norad
    result["satellite"] = SATELLITE_NAMES.get(
        expected_norad, "Unknown"
    )

    print(f"\nAuditing: {result['satellite']} ({expected_norad})")

    # Load one dataset at a time.
    try:
        with file_path.open("r", encoding="utf-8-sig") as file:
            records = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        result["file_status"] = "JSON_ERROR"
        issues.append(f"{filename}: JSON read error: {exc}")
        return result, [], issues

    if not isinstance(records, list):
        result["file_status"] = "INVALID_JSON_STRUCTURE"
        issues.append(
            f"{filename}: Top-level JSON must be a list."
        )
        return result, [], issues

    if not records:
        result["file_status"] = "EMPTY"
        issues.append(f"{filename}: Dataset is empty.")
        return result, [], issues

    result["records"] = len(records)

    gp_ids = Counter()
    tle_pairs = Counter()
    epoch_counts = Counter()
    epoch_tle_pairs = defaultdict(set)
    yearly_counts = Counter()

    epochs = []

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            result["missing_field_records"] += 1
            issues.append(
                f"{filename}: Record {index} is not a JSON object."
            )
            continue

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        missing = [
            field for field in REQUIRED_FIELDS
            if record.get(field) in (None, "")
        ]

        if missing:
            result["missing_field_records"] += 1

        # ----------------------------------------------------
        # NORAD ID
        # ----------------------------------------------------

        if str(record.get("NORAD_CAT_ID", "")) != expected_norad:
            result["norad_mismatches"] += 1

        # ----------------------------------------------------
        # GP_ID duplicates
        # ----------------------------------------------------

        gp_id = record.get("GP_ID")

        if gp_id not in (None, ""):
            gp_ids[str(gp_id)] += 1

        # ----------------------------------------------------
        # EPOCH validation and yearly coverage
        # ----------------------------------------------------

        epoch = None

        try:
            epoch = parse_timestamp(record.get("EPOCH"))

            result["valid_epoch_records"] += 1
            epochs.append(epoch)

            if not START_DATE <= epoch < END_DATE:
                result["out_of_range_epochs"] += 1

            epoch_counts[epoch] += 1

            epoch_tle_pairs[epoch].add((
                record.get("TLE_LINE1"),
                record.get("TLE_LINE2"),
            ))

            yearly_counts[epoch.year] += 1

        except (ValueError, TypeError, OverflowError):
            result["invalid_epoch_records"] += 1

        # ----------------------------------------------------
        # CREATION_DATE validation
        # ----------------------------------------------------

        try:
            parse_timestamp(record.get("CREATION_DATE"))

        except (ValueError, TypeError, OverflowError):
            result["invalid_creation_dates"] += 1

        # ----------------------------------------------------
        # TLE validation
        # ----------------------------------------------------

        line1 = record.get("TLE_LINE1")
        line2 = record.get("TLE_LINE2")

        if (
            not isinstance(line1, str)
            or len(line1) != 69
            or not line1.startswith("1 ")
        ):
            result["invalid_tle_line1"] += 1

        elif not tle_checksum_valid(line1):
            result["bad_tle_line1_checksums"] += 1

        if (
            not isinstance(line2, str)
            or len(line2) != 69
            or not line2.startswith("2 ")
        ):
            result["invalid_tle_line2"] += 1

        elif not tle_checksum_valid(line2):
            result["bad_tle_line2_checksums"] += 1

        # ----------------------------------------------------
        # Exact TLE pair duplicates
        # ----------------------------------------------------

        if (
            isinstance(line1, str)
            and isinstance(line2, str)
        ):
            tle_pairs[(line1, line2)] += 1

    # ========================================================
    # FINAL DATASET STATISTICS
    # ========================================================

    result["duplicate_gp_id_count"] = sum(
        count - 1
        for count in gp_ids.values()
        if count > 1
    )

    result["duplicate_tle_pair_count"] = sum(
        count - 1
        for count in tle_pairs.values()
        if count > 1
    )

    result["repeated_epoch_groups"] = sum(
        count > 1 for count in epoch_counts.values()
    )

    result["conflicting_epoch_groups"] = sum(
        epoch_counts[epoch] > 1 and len(tle_pairs) > 1
        for epoch, tle_pairs in epoch_tle_pairs.items()
    )

    if epochs:
        result["earliest_epoch"] = min(epochs).isoformat()
        result["latest_epoch"] = max(epochs).isoformat()

    result["file_status"] = "READ_OK"

    # --------------------------------------------------------
    # Record issues
    # --------------------------------------------------------

    checks = [
        ("missing required fields", result["missing_field_records"]),
        ("NORAD mismatches", result["norad_mismatches"]),
        ("invalid EPOCH values", result["invalid_epoch_records"]),
        ("out-of-range EPOCH values", result["out_of_range_epochs"]),
        ("invalid CREATION_DATE values", result["invalid_creation_dates"]),
        ("duplicate GP_ID occurrences", result["duplicate_gp_id_count"]),
        ("duplicate TLE pair occurrences", result["duplicate_tle_pair_count"]),
        ("repeated EPOCH groups", result["repeated_epoch_groups"]),
        ("conflicting EPOCH groups", result["conflicting_epoch_groups"]),
        ("invalid TLE_LINE1 values", result["invalid_tle_line1"]),
        ("invalid TLE_LINE2 values", result["invalid_tle_line2"]),
        ("TLE_LINE1 checksum failures", result["bad_tle_line1_checksums"]),
        ("TLE_LINE2 checksum failures", result["bad_tle_line2_checksums"]),
    ]

    for label, count in checks:
        if count:
            issues.append(
                f"{filename}: {count:,} {label}."
            )

    yearly_rows = [
        {
            "satellite": result["satellite"],
            "norad_id": expected_norad,
            "year": year,
            "record_count": count,
        }
        for year, count in sorted(yearly_counts.items())
    ]

    print(
        f"  Records: {result['records']:,} | "
        f"Valid epochs: {result['valid_epoch_records']:,} | "
        f"Years: {len(yearly_counts)}"
    )

    return result, yearly_rows, issues


def _safe_epoch(value):
    """Return a parsed epoch or None."""
    try:
        return parse_timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None

# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 65)
    print("O-RAP: RAW DATASET BATCH AUDIT")
    print("=" * 65)

    if not RAW_DIR.exists():
        raise SystemExit(
            f"Raw data directory not found: {RAW_DIR}"
        )

    files = sorted(
        RAW_DIR.glob("*.json")
    )

    if not files:
        raise SystemExit("No JSON files found in data/raw.")

    print(f"\nJSON files discovered: {len(files)}")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    yearly_rows = []
    all_issues = []

    for file_path in files:

        summary, yearly, issues = audit_file(file_path)

        summaries.append(summary)
        yearly_rows.extend(yearly)
        all_issues.extend(issues)

    # ========================================================
    # WRITE SUMMARY CSV
    # ========================================================

    summary_path = OUTPUT_DIR / "dataset_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summaries)

    # ========================================================
    # WRITE YEARLY COVERAGE CSV
    # ========================================================

    yearly_path = OUTPUT_DIR / "yearly_coverage.csv"

    with yearly_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        fields = [
            "satellite",
            "norad_id",
            "year",
            "record_count",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(yearly_rows)

    # ========================================================
    # WRITE QUALITY REPORT
    # ========================================================

    report_path = OUTPUT_DIR / "data_quality_report.txt"

    total_records = sum(
        row["records"] for row in summaries
    )

    successful_files = sum(
        row["file_status"] == "READ_OK"
        for row in summaries
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("O-RAP RAW DATASET QUALITY REPORT\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Files discovered: {len(files)}\n")
        file.write(f"Files successfully read: {successful_files}\n")
        file.write(f"Total records: {total_records:,}\n")
        file.write(
            f"Requested EPOCH period: {START_DATE.date()} "
            f"to before {END_DATE.date()}\n\n"
        )

        file.write("SATELLITE SUMMARY\n")
        file.write("-" * 65 + "\n")

        for row in summaries:
            file.write(
                f"\n{row['satellite']} "
                f"(NORAD {row['norad_id']})\n"
            )

            file.write(
                f"  File status: {row['file_status']}\n"
            )

            file.write(
                f"  Records: {row['records']:,}\n"
            )

            file.write(
                f"  EPOCH range: {row['earliest_epoch']} "
                f"to {row['latest_epoch']}\n"
            )

            file.write(
                f"  Duplicate GP_ID occurrences: "
                f"{row['duplicate_gp_id_count']:,}\n"
            )

            file.write(
                f"  Repeated EPOCH groups: "
                f"{row['repeated_epoch_groups']:,}\n"
            )

            file.write(
                f"  Conflicting EPOCH groups: "
                f"{row['conflicting_epoch_groups']:,}\n"
            )

        file.write("\n\nDATA QUALITY FINDINGS\n")
        file.write("-" * 65 + "\n")

        if all_issues:
            for issue in all_issues:
                file.write(f"- {issue}\n")
        else:
            file.write("No issues detected by the configured checks.\n")

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print(f"Files discovered       : {len(files)}")
    print(f"Files successfully read: {successful_files}")
    print(f"Total records          : {total_records:,}")
    print(f"Quality findings       : {len(all_issues)}")

    print("\nGenerated files:")
    print(f"1. {summary_path}")
    print(f"2. {yearly_path}")
    print(f"3. {report_path}")

    print("\nRaw datasets were not modified.")


if __name__ == "__main__":
    main()
