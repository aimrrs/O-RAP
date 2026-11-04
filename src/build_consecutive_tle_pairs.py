
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


# ============================================================
# O-RAP: Consecutive TLE Pair Generator
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PAIR_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" /
    "consecutive_tle_pairs.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT / "results" / "consecutive_pair_audit" /
    "pair_summary.csv"
)

REPORT_OUTPUT = (
    PROJECT_ROOT / "results" / "consecutive_pair_audit" /
    "pair_report.txt"
)

RAW_FILES = [
    "aqua_27424_gp_history.json",
    "aura_28376_gp_history.json",
    "cosmo_skymed_1_31598_gp_history.json",
    "dmsp_5d_3_f16_28054_gp_history.json",
    "iss_25544_gp_history.json",
    "landsat_8_39084_gp_history.json",
    "metop_a_29499_gp_history.json",
    "noaa_19_33591_gp_history.json",
    "noaa_20_43013_gp_history.json",
    "radarsat_2_32382_gp_history.json",
    "sentinel_2a_40697_gp_history.json",
    "suomi_npp_37849_gp_history.json",
    "terrasar_x_31698_gp_history.json",
    "terra_25994_gp_history.json",
]


PAIR_FIELDS = [
    "pair_id",
    "satellite_key",
    "satellite_name",
    "norad_id",
    "raw_file",
    "source_record_index",
    "source_gp_id",
    "source_epoch",
    "source_creation_date",
    "reference_record_index",
    "reference_gp_id",
    "reference_epoch",
    "reference_creation_date",
    "epoch_interval_hours",
    "epoch_interval_days",
]


SUMMARY_FIELDS = [
    "satellite_key",
    "satellite_name",
    "norad_id",
    "raw_records",
    "unique_epochs",
    "duplicate_epoch_groups",
    "records_removed_by_epoch_deduplication",
    "consecutive_pairs",
    "min_interval_hours",
    "median_interval_hours",
    "mean_interval_hours",
    "p90_interval_hours",
    "max_interval_hours",
    "interval_under_1h",
    "interval_1_to_6h",
    "interval_6_to_12h",
    "interval_12_to_24h",
    "interval_24_to_48h",
    "interval_over_48h",
]


def parse_timestamp(value):
    """Parse an ISO timestamp and normalize it to UTC."""
    if not value:
        raise ValueError("Missing timestamp")

    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    result = datetime.fromisoformat(text)

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


def format_timestamp(value):
    return value.isoformat().replace("+00:00", "Z")


def percentile(values, fraction):
    """Linearly interpolated percentile."""
    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def load_records(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Support common wrapped JSON structures.
        for key in ("data", "results", "records"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        f"Unsupported JSON structure in {path.name}"
    )


def record_tie_breaker(record):
    """
    Deterministic duplicate-epoch selection:
    1. Latest CREATION_DATE
    2. Highest GP_ID as a string
    3. Latest original record index
    """
    return (
        record["_creation_dt"],
        str(record.get("GP_ID", "")),
        record["_record_index"],
    )


def process_file(filename, pair_writer, pair_counter):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {path}"
        )

    records = load_records(path)

    # Group records by NORAD ID.
    norad_groups = defaultdict(list)

    for index, record in enumerate(records):
        epoch = parse_timestamp(record["EPOCH"])
        creation = parse_timestamp(record["CREATION_DATE"])

        norad_id = str(record["NORAD_CAT_ID"]).strip()

        enriched = dict(record)
        enriched["_record_index"] = index
        enriched["_epoch_dt"] = epoch
        enriched["_creation_dt"] = creation

        norad_groups[norad_id].append(enriched)

    file_summaries = []

    for norad_id, group in sorted(norad_groups.items()):
        # Group records by exact normalized epoch.
        epoch_groups = defaultdict(list)

        for record in group:
            epoch_groups[record["_epoch_dt"]].append(record)

        selected_records = []

        for epoch, duplicate_records in epoch_groups.items():
            selected = max(
                duplicate_records,
                key=record_tie_breaker
            )
            selected_records.append(selected)

        # Chronological order of distinct epochs.
        selected_records.sort(
            key=lambda record: (
                record["_epoch_dt"],
                record["_record_index"],
            )
        )

        intervals = []
        interval_buckets = {
            "under_1h": 0,
            "1_to_6h": 0,
            "6_to_12h": 0,
            "12_to_24h": 0,
            "24_to_48h": 0,
            "over_48h": 0,
        }

        satellite_name = (
            selected_records[-1].get("OBJECT_NAME")
            or filename
        )

        pair_count = 0

        # Pair every epoch with the immediately following epoch.
        for position in range(len(selected_records) - 1):
            source = selected_records[position]
            reference = selected_records[position + 1]

            interval_hours = (
                reference["_epoch_dt"] -
                source["_epoch_dt"]
            ).total_seconds() / 3600.0

            if interval_hours <= 0:
                raise ValueError(
                    f"Non-positive interval in {filename}, "
                    f"NORAD {norad_id}"
                )

            intervals.append(interval_hours)

            if interval_hours < 1:
                interval_buckets["under_1h"] += 1
            elif interval_hours < 6:
                interval_buckets["1_to_6h"] += 1
            elif interval_hours < 12:
                interval_buckets["6_to_12h"] += 1
            elif interval_hours < 24:
                interval_buckets["12_to_24h"] += 1
            elif interval_hours <= 48:
                interval_buckets["24_to_48h"] += 1
            else:
                interval_buckets["over_48h"] += 1

            pair_counter[0] += 1
            pair_count += 1

            pair_writer.writerow({
                "pair_id": pair_counter[0],
                "satellite_key": Path(filename).stem,
                "satellite_name": satellite_name,
                "norad_id": norad_id,
                "raw_file": filename,
                "source_record_index": source["_record_index"],
                "source_gp_id": source.get("GP_ID", ""),
                "source_epoch": format_timestamp(
                    source["_epoch_dt"]
                ),
                "source_creation_date": format_timestamp(
                    source["_creation_dt"]
                ),
                "reference_record_index": reference["_record_index"],
                "reference_gp_id": reference.get("GP_ID", ""),
                "reference_epoch": format_timestamp(
                    reference["_epoch_dt"]
                ),
                "reference_creation_date": format_timestamp(
                    reference["_creation_dt"]
                ),
                "epoch_interval_hours": round(
                    interval_hours, 9
                ),
                "epoch_interval_days": round(
                    interval_hours / 24.0, 9
                ),
            })

        duplicate_groups = sum(
            1 for items in epoch_groups.values()
            if len(items) > 1
        )

        unique_epochs = len(selected_records)
        removed_records = len(group) - unique_epochs

        summary = {
            "satellite_key": Path(filename).stem,
            "satellite_name": satellite_name,
            "norad_id": norad_id,
            "raw_records": len(group),
            "unique_epochs": unique_epochs,
            "duplicate_epoch_groups": duplicate_groups,
            "records_removed_by_epoch_deduplication":
                removed_records,
            "consecutive_pairs": pair_count,
            "min_interval_hours": min(intervals) if intervals else "",
            "median_interval_hours": median(intervals) if intervals else "",
            "mean_interval_hours": (
                sum(intervals) / len(intervals)
                if intervals else ""
            ),
            "p90_interval_hours": (
                percentile(intervals, 0.90)
                if intervals else ""
            ),
            "max_interval_hours": max(intervals) if intervals else "",
            "interval_under_1h": interval_buckets["under_1h"],
            "interval_1_to_6h": interval_buckets["1_to_6h"],
            "interval_6_to_12h": interval_buckets["6_to_12h"],
            "interval_12_to_24h": interval_buckets["12_to_24h"],
            "interval_24_to_48h": interval_buckets["24_to_48h"],
            "interval_over_48h": interval_buckets["over_48h"],
        }

        file_summaries.append(summary)

    return len(records), file_summaries


def main():
    PAIR_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total_raw_records = 0
    pair_counter = [0]

    print("=" * 65)
    print("O-RAP CONSECUTIVE TLE PAIR GENERATION")
    print("=" * 65)

    with PAIR_OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as pair_file:

        pair_writer = csv.DictWriter(
            pair_file,
            fieldnames=PAIR_FIELDS
        )
        pair_writer.writeheader()

        for filename in RAW_FILES:
            print(f"\nProcessing: {filename}")

            raw_count, summaries = process_file(
                filename,
                pair_writer,
                pair_counter
            )

            total_raw_records += raw_count
            all_summaries.extend(summaries)

            for summary in summaries:
                print(
                    f"  {summary['satellite_name']}: "
                    f"{summary['raw_records']:,} records, "
                    f"{summary['unique_epochs']:,} unique epochs, "
                    f"{summary['consecutive_pairs']:,} pairs"
                )

    with SUMMARY_OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:

        writer = csv.DictWriter(
            summary_file,
            fieldnames=SUMMARY_FIELDS
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    total_unique_epochs = sum(
        item["unique_epochs"] for item in all_summaries
    )

    total_duplicate_groups = sum(
        item["duplicate_epoch_groups"]
        for item in all_summaries
    )

    total_removed = sum(
        item["records_removed_by_epoch_deduplication"]
        for item in all_summaries
    )

    total_pairs = sum(
        item["consecutive_pairs"] for item in all_summaries
    )

    report_lines = [
        "O-RAP CONSECUTIVE TLE PAIR AUDIT",
        "=" * 55,
        "",
        "PAIRING METHODOLOGY",
        "- Records grouped by NORAD ID.",
        "- TLE epochs normalized to UTC.",
        "- Duplicate epochs resolved by latest CREATION_DATE.",
        "- Ties resolved using GP_ID, then original record index.",
        "- Records sorted by distinct epoch.",
        "- Each epoch paired with the immediately following epoch.",
        "- No fixed prediction horizon imposed.",
        "",
        "DATASET SUMMARY",
        f"Raw records: {total_raw_records:,}",
        f"Unique epochs: {total_unique_epochs:,}",
        f"Duplicate epoch groups: {total_duplicate_groups:,}",
        f"Records removed by epoch deduplication: {total_removed:,}",
        f"Consecutive pairs generated: {total_pairs:,}",
        f"Satellite/NORAD groups: {len(all_summaries):,}",
        "",
        "INTERVAL BUCKETS",
    ]

    for field, label in [
        ("interval_under_1h", "Under 1 hour"),
        ("interval_1_to_6h", "1 to under 6 hours"),
        ("interval_6_to_12h", "6 to under 12 hours"),
        ("interval_12_to_24h", "12 to under 24 hours"),
        ("interval_24_to_48h", "24 to 48 hours"),
        ("interval_over_48h", "Over 48 hours"),
    ]:
        count = sum(item[field] for item in all_summaries)
        report_lines.append(f"{label}: {count:,}")

    report_lines.extend([
        "",
        "IMPORTANT INTERPRETATION",
        "These pairs measure disagreement between two TLE-derived",
        "SGP4 trajectories at the newer TLE's epoch.",
        "They do not represent independently verified physical",
        "orbit errors.",
        "",
        "Duplicate-epoch resolution retains one representative",
        "record per exact normalized epoch. Original raw files",
        "remain unchanged.",
        "",
        "OUTPUT FILES",
        str(PAIR_OUTPUT),
        str(SUMMARY_OUTPUT),
        str(REPORT_OUTPUT),
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("PAIR GENERATION COMPLETED")
    print("=" * 65)
    print(f"Raw records: {total_raw_records:,}")
    print(f"Unique epochs: {total_unique_epochs:,}")
    print(f"Duplicate epoch groups: {total_duplicate_groups:,}")
    print(f"Records removed: {total_removed:,}")
    print(f"Consecutive pairs: {total_pairs:,}")
    print("\nGenerated files:")
    print(PAIR_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
