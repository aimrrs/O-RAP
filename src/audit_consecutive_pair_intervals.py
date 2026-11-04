
import csv
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


# ============================================================
# O-RAP: Consecutive Pair Interval and Provenance Audit
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PAIR_FILE = (
    ROOT / "data" / "processed" /
    "consecutive_tle_pairs.csv"
)

RAW_DIR = ROOT / "data" / "raw"

OUTPUT_DIR = (
    ROOT / "results" / "consecutive_pair_interval_audit"
)

REPORT_FILE = OUTPUT_DIR / "interval_audit_report.txt"
SHORT_PAIRS_FILE = OUTPUT_DIR / "short_interval_pairs.csv"


def parse_time(value):
    value = str(value).strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "results", "records"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported JSON structure: {path}")


def get_tle_signature(record):
    """
    Exact TLE-line comparison.
    This detects whether both records contain identical TLE data.
    """
    return (
        str(record.get("TLE_LINE1", "")).strip(),
        str(record.get("TLE_LINE2", "")).strip(),
    )


def get_element_signature(record):
    """
    Compare commonly used orbital elements independently
    of the complete TLE line.
    """
    fields = [
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "BSTAR",
        "MEAN_MOTION_DOT",
        "MEAN_MOTION_DDOT",
    ]

    return tuple(
        str(record.get(field, "")).strip()
        for field in fields
    )


def interval_bucket(hours):
    seconds = hours * 3600

    if seconds < 1:
        return "under_1_second"
    elif seconds < 60:
        return "1_second_to_1_minute"
    elif seconds < 300:
        return "1_to_5_minutes"
    elif seconds < 3600:
        return "5_to_60_minutes"
    elif hours < 6:
        return "1_to_under_6_hours"
    elif hours < 12:
        return "6_to_under_12_hours"
    elif hours < 24:
        return "12_to_under_24_hours"
    elif hours <= 48:
        return "24_to_48_hours"
    else:
        return "over_48_hours"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("O-RAP CONSECUTIVE PAIR INTERVAL AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load pair data
    # --------------------------------------------------------

    with PAIR_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        pairs = list(csv.DictReader(f))

    print(f"\nLoaded pairs: {len(pairs):,}")

    # --------------------------------------------------------
    # Load original raw datasets, grouped by filename
    # --------------------------------------------------------

    raw_cache = {}

    def get_raw_record(filename, index):
        if filename not in raw_cache:
            path = RAW_DIR / filename

            if not path.exists():
                raise FileNotFoundError(path)

            raw_cache[filename] = load_json(path)

        records = raw_cache[filename]

        index = int(index)

        if index < 0 or index >= len(records):
            raise IndexError(
                f"Invalid record index {index} in {filename}"
            )

        return records[index]

    # --------------------------------------------------------
    # Counters and statistics
    # --------------------------------------------------------

    buckets = Counter()
    satellite_stats = defaultdict(
        lambda: {
            "count": 0,
            "under_1_second": 0,
            "under_1_minute": 0,
            "under_5_minutes": 0,
            "under_1_hour": 0,
            "identical_tle": 0,
            "identical_elements": 0,
            "provenance_errors": 0,
        }
    )

    intervals = []
    short_pairs = []

    provenance_errors = []
    exact_duplicate_tle_count = 0
    identical_element_count = 0

    checked_pairs = 0

    # --------------------------------------------------------
    # Process every pair
    # --------------------------------------------------------

    for pair in pairs:
        satellite = pair["satellite_name"]
        filename = pair["raw_file"]

        source_index = int(pair["source_record_index"])
        reference_index = int(pair["reference_record_index"])

        source_epoch = parse_time(pair["source_epoch"])
        reference_epoch = parse_time(pair["reference_epoch"])

        actual_hours = (
            reference_epoch - source_epoch
        ).total_seconds() / 3600.0

        actual_seconds = actual_hours * 3600.0

        intervals.append(actual_hours)
        checked_pairs += 1

        bucket = interval_bucket(actual_hours)
        buckets[bucket] += 1

        stats = satellite_stats[satellite]
        stats["count"] += 1

        if actual_seconds < 1:
            stats["under_1_second"] += 1

        if actual_seconds < 60:
            stats["under_1_minute"] += 1

        if actual_seconds < 300:
            stats["under_5_minutes"] += 1

        if actual_seconds < 3600:
            stats["under_1_hour"] += 1

        # Verify raw provenance.
        source_record = get_raw_record(
            filename, source_index
        )

        reference_record = get_raw_record(
            filename, reference_index
        )

        errors = []

        if str(source_record.get("GP_ID", "")) != str(
            pair["source_gp_id"]
        ):
            errors.append("source_gp_id_mismatch")

        if str(reference_record.get("GP_ID", "")) != str(
            pair["reference_gp_id"]
        ):
            errors.append("reference_gp_id_mismatch")

        if parse_time(source_record["EPOCH"]) != source_epoch:
            errors.append("source_epoch_mismatch")

        if parse_time(reference_record["EPOCH"]) != reference_epoch:
            errors.append("reference_epoch_mismatch")

        if str(source_record.get("NORAD_CAT_ID", "")).strip() != str(
            pair["norad_id"]
        ):
            errors.append("source_norad_mismatch")

        if str(reference_record.get("NORAD_CAT_ID", "")).strip() != str(
            pair["norad_id"]
        ):
            errors.append("reference_norad_mismatch")

        if errors:
            stats["provenance_errors"] += 1

            provenance_errors.append({
                "pair_id": pair["pair_id"],
                "satellite": satellite,
                "errors": ";".join(errors),
            })

        # Compare exact TLE lines and orbital elements.
        source_tle = get_tle_signature(source_record)
        reference_tle = get_tle_signature(reference_record)

        identical_tle = source_tle == reference_tle

        source_elements = get_element_signature(source_record)
        reference_elements = get_element_signature(reference_record)

        identical_elements = (
            source_elements == reference_elements
        )

        if identical_tle:
            exact_duplicate_tle_count += 1
            stats["identical_tle"] += 1

        if identical_elements:
            identical_element_count += 1
            stats["identical_elements"] += 1

        # Save all pairs below 5 minutes for inspection.
        if actual_seconds < 300:
            short_pairs.append({
                "pair_id": pair["pair_id"],
                "satellite_name": satellite,
                "norad_id": pair["norad_id"],
                "source_gp_id": pair["source_gp_id"],
                "reference_gp_id": pair["reference_gp_id"],
                "source_epoch": pair["source_epoch"],
                "reference_epoch": pair["reference_epoch"],
                "interval_seconds": round(actual_seconds, 9),
                "identical_tle_lines": identical_tle,
                "identical_orbital_elements": identical_elements,
                "source_record_index": source_index,
                "reference_record_index": reference_index,
                "raw_file": filename,
            })

    # --------------------------------------------------------
    # Save short-interval pairs
    # --------------------------------------------------------

    short_fields = [
        "pair_id",
        "satellite_name",
        "norad_id",
        "source_gp_id",
        "reference_gp_id",
        "source_epoch",
        "reference_epoch",
        "interval_seconds",
        "identical_tle_lines",
        "identical_orbital_elements",
        "source_record_index",
        "reference_record_index",
        "raw_file",
    ]

    with SHORT_PAIRS_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=short_fields
        )
        writer.writeheader()
        writer.writerows(short_pairs)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total = len(pairs)

    report = [
        "O-RAP CONSECUTIVE PAIR INTERVAL AUDIT",
        "=" * 65,
        "",
        "OVERALL",
        f"Total pairs: {total:,}",
        f"Pairs checked: {checked_pairs:,}",
        f"Provenance errors: {len(provenance_errors):,}",
        f"Identical complete TLE lines: {exact_duplicate_tle_count:,}",
        f"Identical orbital element fields: {identical_element_count:,}",
        "",
        "INTERVAL DISTRIBUTION",
    ]

    bucket_labels = [
        ("under_1_second", "Under 1 second"),
        ("1_second_to_1_minute", "1 second to under 1 minute"),
        ("1_to_5_minutes", "1 to under 5 minutes"),
        ("5_to_60_minutes", "5 to under 60 minutes"),
        ("1_to_under_6_hours", "1 to under 6 hours"),
        ("6_to_under_12_hours", "6 to under 12 hours"),
        ("12_to_under_24_hours", "12 to under 24 hours"),
        ("24_to_48_hours", "24 to 48 hours"),
        ("over_48_hours", "Over 48 hours"),
    ]

    for key, label in bucket_labels:
        count = buckets[key]
        percentage = (
            count / total * 100 if total else 0
        )
        report.append(
            f"{label}: {count:,} ({percentage:.4f}%)"
        )

    report.extend([
        "",
        "INTERVAL SUMMARY (HOURS)",
        f"Minimum: {min(intervals):.12f}",
        f"Median: {median(intervals):.9f}",
        f"Maximum: {max(intervals):.9f}",
        "",
        "SATELLITE-LEVEL SHORT INTERVAL AUDIT",
        "Satellite | Pairs | <1 sec | <1 min | <5 min | <1 hour | "
        "Identical TLE | Identical elements | Provenance errors",
    ])

    for satellite, stats in sorted(satellite_stats.items()):
        report.append(
            f"{satellite} | "
            f"{stats['count']:,} | "
            f"{stats['under_1_second']:,} | "
            f"{stats['under_1_minute']:,} | "
            f"{stats['under_5_minutes']:,} | "
            f"{stats['under_1_hour']:,} | "
            f"{stats['identical_tle']:,} | "
            f"{stats['identical_elements']:,} | "
            f"{stats['provenance_errors']:,}"
        )

    report.extend([
        "",
        "INTERPRETATION",
        "- This audit does not remove or modify any pairs.",
        "- Identical TLE lines indicate no change in the complete",
        "  two-line element representation.",
        "- Identical orbital fields compare selected JSON fields.",
        "- Very short intervals require scientific review.",
        "- Provenance errors indicate a mismatch between pair CSV",
        "  references and original raw JSON records.",
        "",
        "OUTPUT",
        str(SHORT_PAIRS_FILE),
    ])

    if provenance_errors:
        report.extend([
            "",
            "PROVENANCE ERROR EXAMPLES",
        ])

        for item in provenance_errors[:50]:
            report.append(str(item))

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERVAL AUDIT COMPLETED")
    print("=" * 70)

    print("\nInterval distribution:")

    for key, label in bucket_labels:
        print(f"{label:30s}: {buckets[key]:,}")

    print("\nTLE comparison:")
    print(f"Identical complete TLE lines: {exact_duplicate_tle_count:,}")
    print(f"Identical orbital elements: {identical_element_count:,}")

    print("\nProvenance:")
    print(f"Errors: {len(provenance_errors):,}")

    print("\nFiles generated:")
    print(REPORT_FILE)
    print(SHORT_PAIRS_FILE)


if __name__ == "__main__":
    main()
