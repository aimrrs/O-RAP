
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday


# ============================================================
# O-RAP: Consecutive TLE SGP4 Validation
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PAIR_FILE = (
    ROOT / "data" / "processed" /
    "consecutive_tle_pairs.csv"
)

RAW_DIR = ROOT / "data" / "raw"

OUTPUT_DIR = (
    ROOT / "results" / "consecutive_sgp4_validation"
)

SAMPLE_FILE = OUTPUT_DIR / "validation_sample.csv"
RESULTS_FILE = OUTPUT_DIR / "sgp4_validation_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "validation_summary.csv"
REPORT_FILE = OUTPUT_DIR / "validation_report.txt"

RANDOM_SEED = 42
RANDOM_PAIRS_PER_STRATUM = 3

# All pairs below this interval are included.
INCLUDE_ALL_BELOW_MINUTES = 5


def parse_timestamp(value):
    value = str(value).strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def interval_category(hours):
    seconds = hours * 3600

    if seconds < 1:
        return "under_1_second"
    elif seconds < 60:
        return "1_second_to_1_minute"
    elif seconds < 300:
        return "1_to_5_minutes"
    elif hours < 1:
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


def load_raw_records(filename, cache):
    if filename not in cache:
        path = RAW_DIR / filename

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("data", "results", "records"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            raise ValueError(
                f"Unsupported JSON structure: {filename}"
            )

        cache[filename] = data

    return cache[filename]


def get_position(satrec, timestamp):
    """
    Propagate a TLE to an exact UTC datetime using SGP4.
    Returns error code, TEME position (km), velocity (km/s).
    """
    jd, fr = jday(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second
        + timestamp.microsecond / 1_000_000.0,
    )

    error_code, position, velocity = satrec.sgp4(jd, fr)

    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    return error_code, position, velocity


def vector_norm(vector):
    return float(np.linalg.norm(vector))


def select_validation_sample(pairs):
    rng = random.Random(RANDOM_SEED)

    strata = defaultdict(list)
    selected = {}

    for pair in pairs:
        hours = float(pair["epoch_interval_hours"])
        category = interval_category(hours)

        key = (
            pair["satellite_key"],
            category,
        )

        strata[key].append(pair)

        # Include every pair below five minutes.
        if hours * 3600 < INCLUDE_ALL_BELOW_MINUTES * 60:
            selected[pair["pair_id"]] = pair

    # Add a reproducible random sample from every stratum.
    for key, records in strata.items():
        candidates = [
            record for record in records
            if record["pair_id"] not in selected
        ]

        count = min(
            RANDOM_PAIRS_PER_STRATUM,
            len(candidates)
        )

        if count:
            for record in rng.sample(candidates, count):
                selected[record["pair_id"]] = record

    return list(selected.values()), strata


def validate_pair(pair, raw_cache):
    filename = pair["raw_file"]

    records = load_raw_records(filename, raw_cache)

    source_index = int(pair["source_record_index"])
    reference_index = int(pair["reference_record_index"])

    source_record = records[source_index]
    reference_record = records[reference_index]

    # Confirm that the selected raw records still match the pair.
    if str(source_record.get("GP_ID", "")) != str(
        pair["source_gp_id"]
    ):
        raise ValueError(
            f"Source GP_ID mismatch: pair {pair['pair_id']}"
        )

    if str(reference_record.get("GP_ID", "")) != str(
        pair["reference_gp_id"]
    ):
        raise ValueError(
            f"Reference GP_ID mismatch: pair {pair['pair_id']}"
        )

    if str(source_record.get("NORAD_CAT_ID", "")).strip() != str(
        pair["norad_id"]
    ):
        raise ValueError(
            f"Source NORAD mismatch: pair {pair['pair_id']}"
        )

    if str(reference_record.get("NORAD_CAT_ID", "")).strip() != str(
        pair["norad_id"]
    ):
        raise ValueError(
            f"Reference NORAD mismatch: pair {pair['pair_id']}"
        )

    source_epoch = parse_timestamp(pair["source_epoch"])
    reference_epoch = parse_timestamp(pair["reference_epoch"])

    if parse_timestamp(source_record["EPOCH"]) != source_epoch:
        raise ValueError(
            f"Source epoch mismatch: pair {pair['pair_id']}"
        )

    if parse_timestamp(reference_record["EPOCH"]) != reference_epoch:
        raise ValueError(
            f"Reference epoch mismatch: pair {pair['pair_id']}"
        )

    source_line1 = source_record["TLE_LINE1"].strip()
    source_line2 = source_record["TLE_LINE2"].strip()

    reference_line1 = reference_record["TLE_LINE1"].strip()
    reference_line2 = reference_record["TLE_LINE2"].strip()

    source_sat = Satrec.twoline2rv(
        source_line1,
        source_line2
    )

    reference_sat = Satrec.twoline2rv(
        reference_line1,
        reference_line2
    )

    # Both satellites are evaluated at the reference TLE epoch.
    source_error, source_position, source_velocity = (
        get_position(source_sat, reference_epoch)
    )

    reference_error, reference_position, reference_velocity = (
        get_position(reference_sat, reference_epoch)
    )

    result = {
        "pair_id": pair["pair_id"],
        "satellite_name": pair["satellite_name"],
        "satellite_key": pair["satellite_key"],
        "norad_id": pair["norad_id"],
        "requested_interval_hours": pair["epoch_interval_hours"],
        "interval_category": interval_category(
            float(pair["epoch_interval_hours"])
        ),
        "source_gp_id": pair["source_gp_id"],
        "reference_gp_id": pair["reference_gp_id"],
        "source_epoch": pair["source_epoch"],
        "reference_epoch": pair["reference_epoch"],
        "source_sgp4_error": source_error,
        "reference_sgp4_error": reference_error,
        "source_position_norm_km": "",
        "reference_position_norm_km": "",
        "source_velocity_norm_km_s": "",
        "reference_velocity_norm_km_s": "",
        "radial_difference_km": "",
        "along_track_difference_km": "",
        "cross_track_difference_km": "",
        "disagreement_km": "",
        "unreliable_1km": "",
        "finite_positions": False,
        "plausible_leo_position_norms": False,
        "failure_reason": "",
    }

    if source_error != 0 or reference_error != 0:
        result["failure_reason"] = (
            f"SGP4 error: source={source_error}, "
            f"reference={reference_error}"
        )
        return result

    all_vectors = [
        source_position,
        reference_position,
        source_velocity,
        reference_velocity,
    ]

    if not all(np.all(np.isfinite(v)) for v in all_vectors):
        result["failure_reason"] = "Non-finite SGP4 output"
        return result

    source_norm = vector_norm(source_position)
    reference_norm = vector_norm(reference_position)

    source_speed = vector_norm(source_velocity)
    reference_speed = vector_norm(reference_velocity)

    difference = source_position - reference_position
    disagreement = vector_norm(difference)

    # Local radial / along-track / cross-track decomposition
    # based on the reference TLE state at its own epoch.
    radial_unit = reference_position / reference_norm

    angular_momentum = np.cross(
        reference_position,
        reference_velocity
    )

    angular_momentum_norm = vector_norm(angular_momentum)

    if angular_momentum_norm == 0:
        result["failure_reason"] = (
            "Cannot calculate orbital frame"
        )
        return result

    cross_track_unit = (
        angular_momentum / angular_momentum_norm
    )

    along_track_unit = np.cross(
        cross_track_unit,
        radial_unit
    )

    radial_difference = float(
        np.dot(difference, radial_unit)
    )

    along_difference = float(
        np.dot(difference, along_track_unit)
    )

    cross_difference = float(
        np.dot(difference, cross_track_unit)
    )

    result.update({
        "source_position_norm_km": source_norm,
        "reference_position_norm_km": reference_norm,
        "source_velocity_norm_km_s": source_speed,
        "reference_velocity_norm_km_s": reference_speed,
        "radial_difference_km": radial_difference,
        "along_track_difference_km": along_difference,
        "cross_track_difference_km": cross_difference,
        "disagreement_km": disagreement,
        "unreliable_1km": int(disagreement > 1.0),
        "finite_positions": True,
        "plausible_leo_position_norms": (
            6000 <= source_norm <= 8000
            and 6000 <= reference_norm <= 8000
        ),
        "failure_reason": "",
    })

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("O-RAP CONSECUTIVE TLE SGP4 VALIDATION")
    print("=" * 70)

    with PAIR_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        pairs = list(csv.DictReader(f))

    print(f"\nTotal consecutive pairs: {len(pairs):,}")

    sample, strata = select_validation_sample(pairs)

    print(f"Selected validation sample: {len(sample):,}")
    print(
        "Pairs below five minutes included:",
        sum(
            float(p["epoch_interval_hours"]) * 3600 < 300
            for p in sample
        )
    )

    with SAMPLE_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(sample[0].keys())
        )
        writer.writeheader()
        writer.writerows(sample)

    raw_cache = {}
    results = []

    for index, pair in enumerate(sample, start=1):
        if index % 100 == 0:
            print(
                f"Validated {index:,}/{len(sample):,}"
            )

        try:
            result = validate_pair(pair, raw_cache)
        except Exception as exc:
            result = {
                "pair_id": pair["pair_id"],
                "satellite_name": pair["satellite_name"],
                "satellite_key": pair["satellite_key"],
                "norad_id": pair["norad_id"],
                "requested_interval_hours": pair[
                    "epoch_interval_hours"
                ],
                "interval_category": interval_category(
                    float(pair["epoch_interval_hours"])
                ),
                "source_gp_id": pair["source_gp_id"],
                "reference_gp_id": pair["reference_gp_id"],
                "source_epoch": pair["source_epoch"],
                "reference_epoch": pair["reference_epoch"],
                "source_sgp4_error": "",
                "reference_sgp4_error": "",
                "source_position_norm_km": "",
                "reference_position_norm_km": "",
                "source_velocity_norm_km_s": "",
                "reference_velocity_norm_km_s": "",
                "radial_difference_km": "",
                "along_track_difference_km": "",
                "cross_track_difference_km": "",
                "disagreement_km": "",
                "unreliable_1km": "",
                "finite_positions": False,
                "plausible_leo_position_norms": False,
                "failure_reason": str(exc),
            }

        results.append(result)

    fields = list(results[0].keys())

    with RESULTS_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    successful = [
        r for r in results
        if r["failure_reason"] == ""
        and r["disagreement_km"] != ""
    ]

    failed = [
        r for r in results
        if r["failure_reason"] != ""
    ]

    category_stats = defaultdict(list)
    satellite_stats = defaultdict(list)

    for result in successful:
        category_stats[
            result["interval_category"]
        ].append(float(result["disagreement_km"]))

        satellite_stats[
            result["satellite_name"]
        ].append(float(result["disagreement_km"]))

    summary_rows = []

    for category, values in sorted(category_stats.items()):
        ordered = sorted(values)

        summary_rows.append({
            "group_type": "interval_category",
            "group_name": category,
            "successful_pairs": len(values),
            "positive_pairs": sum(v > 1.0 for v in values),
            "positive_rate": sum(v > 1.0 for v in values) / len(values),
            "mean_disagreement_km": sum(values) / len(values),
            "median_disagreement_km": float(np.median(values)),
            "p90_disagreement_km": float(np.percentile(values, 90)),
            "p99_disagreement_km": float(np.percentile(values, 99)),
            "maximum_disagreement_km": max(values),
        })

    for satellite, values in sorted(satellite_stats.items()):
        summary_rows.append({
            "group_type": "satellite",
            "group_name": satellite,
            "successful_pairs": len(values),
            "positive_pairs": sum(v > 1.0 for v in values),
            "positive_rate": sum(v > 1.0 for v in values) / len(values),
            "mean_disagreement_km": sum(values) / len(values),
            "median_disagreement_km": float(np.median(values)),
            "p90_disagreement_km": float(np.percentile(values, 90)),
            "p99_disagreement_km": float(np.percentile(values, 99)),
            "maximum_disagreement_km": max(values),
        })

    summary_fields = [
        "group_type",
        "group_name",
        "successful_pairs",
        "positive_pairs",
        "positive_rate",
        "mean_disagreement_km",
        "median_disagreement_km",
        "p90_disagreement_km",
        "p99_disagreement_km",
        "maximum_disagreement_km",
    ]

    with SUMMARY_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=summary_fields
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    disagreements = [
        float(r["disagreement_km"])
        for r in successful
    ]

    positive_count = sum(
        value > 1.0 for value in disagreements
    )

    report = [
        "O-RAP CONSECUTIVE TLE SGP4 VALIDATION",
        "=" * 65,
        "",
        "SAMPLE DESIGN",
        f"Total dataset pairs: {len(pairs):,}",
        f"Validation sample: {len(sample):,}",
        f"Random seed: {RANDOM_SEED}",
        f"Random pairs per satellite/category: {RANDOM_PAIRS_PER_STRATUM}",
        "All pairs below five minutes included: Yes",
        "",
        "PROPAGATION RESULTS",
        f"Successful pairs: {len(successful):,}",
        f"Failed pairs: {len(failed):,}",
        f"Successful propagation rate: "
        f"{len(successful) / len(sample) * 100:.4f}%",
        "",
        "POSITION DISAGREEMENT",
        f"Positive pairs (>1 km): {positive_count:,}",
        f"Negative pairs (<=1 km): "
        f"{len(successful) - positive_count:,}",
        f"Positive rate: "
        f"{positive_count / len(successful) * 100:.4f}%"
        if successful else "Positive rate: N/A",
        f"Mean disagreement (km): "
        f"{np.mean(disagreements):.9f}"
        if disagreements else "Mean disagreement: N/A",
        f"Median disagreement (km): "
        f"{np.median(disagreements):.9f}"
        if disagreements else "Median disagreement: N/A",
        f"Maximum disagreement (km): "
        f"{max(disagreements):.9f}"
        if disagreements else "Maximum disagreement: N/A",
        "",
        "POSITION SANITY CHECK",
        f"Finite successful outputs: {len(successful):,}",
        "LEO position norm check range: 6000 to 8000 km",
        "This is a diagnostic range, not a filtering rule.",
        "",
        "FAILURE DETAILS",
    ]

    if failed:
        for item in failed[:30]:
            report.append(
                f"Pair {item['pair_id']}: "
                f"{item['failure_reason']}"
            )
    else:
        report.append("No failures in validation sample.")

    report.extend([
        "",
        "TOP 20 DISAGREEMENTS",
    ])

    for item in sorted(
        successful,
        key=lambda r: float(r["disagreement_km"]),
        reverse=True
    )[:20]:
        report.append(
            f"Pair {item['pair_id']} | "
            f"{item['satellite_name']} | "
            f"{item['interval_category']} | "
            f"{float(item['disagreement_km']):.6f} km | "
            f"source={item['source_gp_id']} | "
            f"reference={item['reference_gp_id']}"
        )

    report.extend([
        "",
        "SCIENTIFIC INTERPRETATION",
        "Both TLE-derived SGP4 states are evaluated at the",
        "newer TLE's epoch in the TEME frame.",
        "The disagreement is a TLE-to-TLE position proxy,",
        "not independently verified physical orbit error.",
        "No pairs have been removed based on disagreement.",
        "",
        "OUTPUT FILES",
        str(SAMPLE_FILE),
        str(RESULTS_FILE),
        str(SUMMARY_FILE),
        str(REPORT_FILE),
    ])

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)
    print(f"Sample size: {len(sample):,}")
    print(f"Successful: {len(successful):,}")
    print(f"Failed: {len(failed):,}")
    print(f"Positive (>1 km): {positive_count:,}")

    if disagreements:
        print(
            f"Median disagreement: "
            f"{np.median(disagreements):.6f} km"
        )
        print(
            f"Maximum disagreement: "
            f"{max(disagreements):.6f} km"
        )

    print("\nGenerated files:")
    print(SAMPLE_FILE)
    print(RESULTS_FILE)
    print(SUMMARY_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
