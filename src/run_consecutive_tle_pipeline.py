
import json
import math
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "results" / "consecutive_tle_pipeline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_FILE = OUTPUT_DIR / "consecutive_tle_labeled_dataset.csv"
SATELLITE_SUMMARY_FILE = OUTPUT_DIR / "sgp4_quality_summary.csv"
INTERVAL_SUMMARY_FILE = OUTPUT_DIR / "interval_quality_summary.csv"
ANOMALY_FILE = OUTPUT_DIR / "anomaly_pairs.csv"

LABEL_THRESHOLD_KM = 1.0

# These are flags for investigation, not exclusion criteria.
SHORT_INTERVAL_SECONDS = 60.0
LONG_INTERVAL_HOURS = 48.0
LARGE_DISAGREEMENT_KM = 10.0
EXTREME_DISAGREEMENT_KM = 100.0

# TLE epoch is encoded to 1e-8 day, approximately 0.864 ms.
# Allow a small tolerance when comparing the encoded and JSON epochs.
TLE_EPOCH_TOLERANCE_SECONDS = 0.002

# Previously independently reproduced cases.
KNOWN_REPRODUCTION_PAIRS = {
    ("86185510", "86185505"): 22.468937685080,
    ("109400906", "109398766"): 25.546845635782,
}

# Additional previously investigated outlier pairs.
KNOWN_OUTLIER_PAIRS = {
    ("82046287", "82058366"),
    ("95872723", "95916775"),
    ("97085453", "97085599"),
    ("91798876", "91709247"),
    ("99547653", "100195142"),
    ("57291659", "57291699"),
    ("99211973", "99212495"),
    ("93142454", "93142414"),
    ("109398766", "109398755"),
    ("98961385", "98961576"),
    ("86185510", "86185505"),
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_datetime(value):
    """Parse an ISO timestamp and normalize it to UTC."""
    if value is None or pd.isna(value):
        raise ValueError("Missing timestamp")

    dt = datetime.fromisoformat(
        str(value).strip().replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def load_json_records(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "records", "results"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unexpected JSON structure: {path}")


def checksum_valid(line):
    if not isinstance(line, str) or len(line) != 69:
        return False

    total = 0

    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1

    return str(total % 10) == line[68]


def tle_satellite_number(line):
    if not isinstance(line, str) or len(line) < 7:
        return None

    return line[2:7].strip()


def tle_encoded_epoch(line1):
    """Decode the epoch in TLE Line 1."""
    if not isinstance(line1, str) or len(line1) < 32:
        raise ValueError("TLE Line 1 is too short")

    year_short = int(line1[18:20])
    day_of_year = float(line1[20:32])

    year = 2000 + year_short if year_short < 57 else 1900 + year_short

    return (
        datetime(year, 1, 1, tzinfo=timezone.utc)
        + timedelta(days=day_of_year - 1.0)
    )


def vector_norm(vector):
    return math.sqrt(sum(float(x) ** 2 for x in vector))


def vector_subtract(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def vector_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def vector_scale(a, scale):
    return [float(x) * scale for x in a]


def vector_unit(a):
    length = vector_norm(a)

    if length == 0 or not math.isfinite(length):
        return None

    return vector_scale(a, 1.0 / length)


def rtn_components(reference_position, reference_velocity, difference):
    """
    Resolve SOURCE - REFERENCE position difference into the
    reference orbit's radial, along-track and cross-track axes.
    """
    radial = vector_unit(reference_position)

    angular_momentum = vector_cross(
        reference_position,
        reference_velocity,
    )
    cross_track = vector_unit(angular_momentum)

    if radial is None or cross_track is None:
        return None, None, None

    along_track = vector_unit(
        vector_cross(cross_track, radial)
    )

    if along_track is None:
        return None, None, None

    return (
        vector_dot(difference, radial),
        vector_dot(difference, along_track),
        vector_dot(difference, cross_track),
    )


def propagate_to_epoch(line1, line2, target_time):
    """Propagate one TLE to the supplied UTC timestamp."""
    satellite = Satrec.twoline2rv(line1, line2)

    jd, fr = jday(
        target_time.year,
        target_time.month,
        target_time.day,
        target_time.hour,
        target_time.minute,
        target_time.second
        + target_time.microsecond / 1_000_000.0,
    )

    error_code, position, velocity = satellite.sgp4(jd, fr)

    return {
        "error_code": int(error_code),
        "position": [float(x) for x in position],
        "velocity": [float(x) for x in velocity],
    }


def finite_vector(vector):
    return (
        vector is not None
        and len(vector) == 3
        and all(math.isfinite(float(x)) for x in vector)
    )


def interval_category(seconds):
    if seconds < 1:
        return "under_1_second"
    if seconds < 60:
        return "1_second_to_under_1_minute"
    if seconds < 300:
        return "1_to_5_minutes"
    if seconds < 3600:
        return "5_to_60_minutes"
    if seconds < 6 * 3600:
        return "1_to_under_6_hours"
    if seconds < 12 * 3600:
        return "6_to_under_12_hours"
    if seconds < 24 * 3600:
        return "12_to_under_24_hours"
    if seconds <= 48 * 3600:
        return "24_to_48_hours"
    return "over_48_hours"


def safe_string(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


# ============================================================
# TLE RECORD VALIDATION
# ============================================================

def validate_tle_record(record, expected_norad):
    """
    Return a parsed SGP4 satellite object and record-level
    quality information.
    """
    result = {
        "valid": False,
        "reason": "",
        "line1_checksum_ok": False,
        "line2_checksum_ok": False,
        "satellite_number_ok": False,
        "epoch_match": False,
        "satellite": None,
        "line1": "",
        "line2": "",
    }

    line1 = safe_string(record.get("TLE_LINE1"))
    line2 = safe_string(record.get("TLE_LINE2"))

    result["line1"] = line1
    result["line2"] = line2

    result["line1_checksum_ok"] = checksum_valid(line1)
    result["line2_checksum_ok"] = checksum_valid(line2)

    if not result["line1_checksum_ok"]:
        result["reason"] = "invalid_line1_checksum"
        return result

    if not result["line2_checksum_ok"]:
        result["reason"] = "invalid_line2_checksum"
        return result

    line1_norad = tle_satellite_number(line1)
    line2_norad = tle_satellite_number(line2)

    result["satellite_number_ok"] = (
        line1_norad == line2_norad
        and line1_norad == str(expected_norad)
    )

    if not result["satellite_number_ok"]:
        result["reason"] = "tle_satellite_number_mismatch"
        return result

    try:
        json_epoch = parse_datetime(record.get("EPOCH"))
        encoded_epoch = tle_encoded_epoch(line1)

        epoch_difference = abs(
            (json_epoch - encoded_epoch).total_seconds()
        )

        result["epoch_difference_seconds"] = epoch_difference

        result["epoch_match"] = (
            epoch_difference <= TLE_EPOCH_TOLERANCE_SECONDS
        )

        if not result["epoch_match"]:
            result["reason"] = "tle_json_epoch_mismatch"
            return result

        satellite = Satrec.twoline2rv(line1, line2)

        result["satellite"] = satellite
        result["valid"] = True
        result["reason"] = "valid"

    except Exception as exc:
        result["reason"] = "tle_parse_error: " + str(exc)

    return result


# ============================================================
# DATA LOADING AND CONSECUTIVE PAIR CONSTRUCTION
# ============================================================

def load_and_pair_all_datasets():
    raw_files = sorted(RAW_DIR.glob("*_gp_history.json"))

    if not raw_files:
        raise FileNotFoundError(
            f"No *_gp_history.json files found in {RAW_DIR}"
        )

    print("\nSTAGE 1: RAW DATA LOADING AND PAIR CONSTRUCTION")
    print("-" * 72)

    all_pairs = []
    total_raw_records = 0
    total_unique_epochs = 0
    total_duplicate_groups = 0
    total_removed_duplicates = 0

    for raw_file in raw_files:
        records = load_json_records(raw_file)
        total_raw_records += len(records)

        if not records:
            print(f"WARNING: Empty dataset: {raw_file.name}")
            continue

        norad_values = {
            safe_string(record.get("NORAD_CAT_ID"))
            for record in records
            if safe_string(record.get("NORAD_CAT_ID"))
        }

        if len(norad_values) != 1:
            raise ValueError(
                f"NORAD inconsistency in {raw_file.name}: {norad_values}"
            )

        norad_id = next(iter(norad_values))
        satellite_name = safe_string(
            records[0].get("OBJECT_NAME")
        ) or raw_file.stem

        normalized = []

        for index, record in enumerate(records):
            try:
                epoch = parse_datetime(record.get("EPOCH"))
                creation_value = record.get("CREATION_DATE")

                try:
                    creation_date = parse_datetime(creation_value)
                    creation_sort = creation_date.timestamp()
                except Exception:
                    creation_date = None
                    creation_sort = float("-inf")

                normalized.append({
                    "record": record,
                    "record_index": index,
                    "epoch": epoch,
                    "creation_date": creation_date,
                    "creation_sort": creation_sort,
                    "gp_id": safe_string(record.get("GP_ID")),
                })

            except Exception as exc:
                raise ValueError(
                    f"Invalid epoch in {raw_file.name}, "
                    f"record {index}: {exc}"
                )

        # Exact epoch duplicate groups.
        epoch_groups = {}

        for item in normalized:
            epoch_groups.setdefault(item["epoch"], []).append(item)

        duplicate_groups = sum(
            1 for group in epoch_groups.values() if len(group) > 1
        )

        # Deterministic duplicate selection:
        # latest CREATION_DATE, then highest GP_ID string,
        # then latest original record index.
        selected = []

        for epoch, group in epoch_groups.items():
            chosen = max(
                group,
                key=lambda item: (
                    item["creation_sort"],
                    item["gp_id"],
                    item["record_index"],
                ),
            )
            selected.append(chosen)

        selected.sort(key=lambda item: item["epoch"])

        total_unique_epochs += len(selected)
        total_duplicate_groups += duplicate_groups
        total_removed_duplicates += len(records) - len(selected)

        satellite_pair_count = max(0, len(selected) - 1)

        for i in range(satellite_pair_count):
            source = selected[i]
            reference = selected[i + 1]

            interval_seconds = (
                reference["epoch"] - source["epoch"]
            ).total_seconds()

            if interval_seconds <= 0:
                raise ValueError(
                    f"Non-positive interval in {raw_file.name}: "
                    f"{source['epoch']} -> {reference['epoch']}"
                )

            all_pairs.append({
                "pair_id": len(all_pairs),
                "satellite_key": raw_file.stem,
                "satellite_name": satellite_name,
                "norad_id": norad_id,
                "raw_file": raw_file.name,
                "source_record_index": source["record_index"],
                "reference_record_index": reference["record_index"],
                "source_record": source["record"],
                "reference_record": reference["record"],
                "source_gp_id": source["gp_id"],
                "reference_gp_id": reference["gp_id"],
                "source_epoch": source["epoch"],
                "reference_epoch": reference["epoch"],
                "source_creation_date": source["creation_date"],
                "reference_creation_date": reference["creation_date"],
                "interval_seconds": interval_seconds,
                "interval_hours": interval_seconds / 3600.0,
            })

        print(
            f"{satellite_name:<24} "
            f"raw={len(records):>8,}  "
            f"unique_epochs={len(selected):>8,}  "
            f"pairs={satellite_pair_count:>8,}"
        )

    print("-" * 72)
    print(f"Raw records:                 {total_raw_records:,}")
    print(f"Unique epochs:               {total_unique_epochs:,}")
    print(f"Duplicate epoch groups:      {total_duplicate_groups:,}")
    print(f"Records removed by dedup:    {total_removed_duplicates:,}")
    print(f"Consecutive pairs:           {len(all_pairs):,}")
    print(f"Satellite datasets:          {len(raw_files)}")

    expected_pairs = total_unique_epochs - len(raw_files)

    print(f"Expected pair count:         {expected_pairs:,}")

    if len(all_pairs) != expected_pairs:
        raise RuntimeError(
            "Pair count verification failed: "
            f"actual={len(all_pairs)}, expected={expected_pairs}"
        )

    print("Pair count verification:     PASS")

    return all_pairs


# ============================================================
# KNOWN PAIR REPRODUCTION CHECK
# ============================================================

def verify_known_reproductions(results):
    print("\nSTAGE 2: KNOWN PAIR REPRODUCTION CHECK")
    print("-" * 72)

    lookup = {
        (row["source_gp_id"], row["reference_gp_id"]): row
        for row in results
    }

    all_passed = True

    for gp_pair, expected_value in KNOWN_REPRODUCTION_PAIRS.items():
        row = lookup.get(gp_pair)

        if row is None:
            print(f"{gp_pair}: NOT FOUND")
            all_passed = False
            continue

        actual = row.get("disagreement_km")

        if actual is None or not math.isfinite(actual):
            print(f"{gp_pair}: NO VALID DISAGREEMENT")
            all_passed = False
            continue

        difference = abs(actual - expected_value)
        passed = difference < 0.000001

        print(
            f"GP IDs {gp_pair[0]} -> {gp_pair[1]} | "
            f"expected={expected_value:.12f} km | "
            f"actual={actual:.12f} km | "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_passed = False

    print(
        "Known reproduction result:",
        "PASS" if all_passed else "REVIEW REQUIRED",
    )

    return all_passed


# ============================================================
# FULL SGP4 PROPAGATION AND LABEL GENERATION
# ============================================================

def process_pair(pair):
    source_record = pair["source_record"]
    reference_record = pair["reference_record"]

    source_check = validate_tle_record(
        source_record,
        pair["norad_id"],
    )
    reference_check = validate_tle_record(
        reference_record,
        pair["norad_id"],
    )

    row = {
        key: value
        for key, value in pair.items()
        if key not in ("source_record", "reference_record")
    }

    row["source_tle_valid"] = source_check["valid"]
    row["reference_tle_valid"] = reference_check["valid"]

    row["source_tle_reason"] = source_check["reason"]
    row["reference_tle_reason"] = reference_check["reason"]

    row["source_line1_checksum_ok"] = (
        source_check["line1_checksum_ok"]
    )
    row["source_line2_checksum_ok"] = (
        source_check["line2_checksum_ok"]
    )
    row["reference_line1_checksum_ok"] = (
        reference_check["line1_checksum_ok"]
    )
    row["reference_line2_checksum_ok"] = (
        reference_check["line2_checksum_ok"]
    )

    row["source_tle_epoch_match"] = source_check["epoch_match"]
    row["reference_tle_epoch_match"] = reference_check["epoch_match"]

    row["source_sgp4_error"] = None
    row["reference_sgp4_error"] = None

    row["source_position_norm_km"] = None
    row["reference_position_norm_km"] = None
    row["source_velocity_norm_km_s"] = None
    row["reference_velocity_norm_km_s"] = None

    row["dx_km"] = None
    row["dy_km"] = None
    row["dz_km"] = None

    row["radial_difference_km"] = None
    row["along_track_difference_km"] = None
    row["cross_track_difference_km"] = None

    row["disagreement_km"] = None
    row["label_1km"] = None

    row["position_norm_flag"] = False
    row["large_disagreement_flag"] = False
    row["extreme_disagreement_flag"] = False
    row["short_interval_flag"] = (
        pair["interval_seconds"] < SHORT_INTERVAL_SECONDS
    )
    row["long_interval_flag"] = (
        pair["interval_hours"] > LONG_INTERVAL_HOURS
    )

    row["known_outlier_pair"] = (
        pair["source_gp_id"],
        pair["reference_gp_id"],
    ) in KNOWN_OUTLIER_PAIRS

    row["status"] = "not_processed"

    # Integrity gate.
    if not source_check["valid"]:
        row["status"] = "invalid_source_tle"
        return row

    if not reference_check["valid"]:
        row["status"] = "invalid_reference_tle"
        return row

    try:
        # The target is the newer/reference TLE's JSON epoch.
        target_time = pair["reference_epoch"]

        source_result = propagate_to_epoch(
            source_check["line1"],
            source_check["line2"],
            target_time,
        )

        reference_result = propagate_to_epoch(
            reference_check["line1"],
            reference_check["line2"],
            target_time,
        )

        row["source_sgp4_error"] = source_result["error_code"]
        row["reference_sgp4_error"] = reference_result["error_code"]

        if source_result["error_code"] != 0:
            row["status"] = "source_sgp4_error"
            return row

        if reference_result["error_code"] != 0:
            row["status"] = "reference_sgp4_error"
            return row

        source_position = source_result["position"]
        reference_position = reference_result["position"]

        source_velocity = source_result["velocity"]
        reference_velocity = reference_result["velocity"]

        if not finite_vector(source_position):
            row["status"] = "nonfinite_source_position"
            return row

        if not finite_vector(reference_position):
            row["status"] = "nonfinite_reference_position"
            return row

        if not finite_vector(source_velocity):
            row["status"] = "nonfinite_source_velocity"
            return row

        if not finite_vector(reference_velocity):
            row["status"] = "nonfinite_reference_velocity"
            return row

        source_position_norm = vector_norm(source_position)
        reference_position_norm = vector_norm(reference_position)

        source_velocity_norm = vector_norm(source_velocity)
        reference_velocity_norm = vector_norm(reference_velocity)

        row["source_position_norm_km"] = source_position_norm
        row["reference_position_norm_km"] = reference_position_norm

        row["source_velocity_norm_km_s"] = source_velocity_norm
        row["reference_velocity_norm_km_s"] = reference_velocity_norm

        # Broad LEO plausibility flag only; not a deletion rule.
        if not (
            6000.0 <= source_position_norm <= 8000.0
            and 6000.0 <= reference_position_norm <= 8000.0
        ):
            row["position_norm_flag"] = True

        difference = vector_subtract(
            source_position,
            reference_position,
        )

        disagreement = vector_norm(difference)

        if not math.isfinite(disagreement):
            row["status"] = "nonfinite_disagreement"
            return row

        row["dx_km"] = difference[0]
        row["dy_km"] = difference[1]
        row["dz_km"] = difference[2]

        radial, along, cross = rtn_components(
            reference_position,
            reference_velocity,
            difference,
        )

        row["radial_difference_km"] = radial
        row["along_track_difference_km"] = along
        row["cross_track_difference_km"] = cross

        row["disagreement_km"] = disagreement
        row["label_1km"] = int(
            disagreement > LABEL_THRESHOLD_KM
        )

        row["large_disagreement_flag"] = (
            disagreement > LARGE_DISAGREEMENT_KM
        )

        row["extreme_disagreement_flag"] = (
            disagreement > EXTREME_DISAGREEMENT_KM
        )

        row["status"] = "labeled"

    except Exception as exc:
        row["status"] = "processing_exception: " + str(exc)

    return row


# ============================================================
# SUMMARY GENERATION
# ============================================================

def make_interval_summary(df):
    valid = df[df["status"] == "labeled"].copy()

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("interval_category", dropna=False)
        .agg(
            pair_count=("pair_id", "count"),
            positive_count=("label_1km", "sum"),
            median_disagreement_km=("disagreement_km", "median"),
            mean_disagreement_km=("disagreement_km", "mean"),
            p95_disagreement_km=(
                "disagreement_km",
                lambda x: x.quantile(0.95),
            ),
            maximum_disagreement_km=("disagreement_km", "max"),
        )
        .reset_index()
    )

    summary["positive_rate_percent"] = (
        100.0 * summary["positive_count"] / summary["pair_count"]
    )

    return summary


def make_satellite_summary(df):
    summary = (
        df.groupby(
            ["satellite_name", "norad_id"],
            dropna=False,
        )
        .agg(
            total_pairs=("pair_id", "count"),
            labeled_pairs=("label_1km", "count"),
            propagation_status_labeled=(
                "status",
                lambda x: int((x == "labeled").sum()),
            ),
            positive_count=("label_1km", "sum"),
            mean_disagreement_km=("disagreement_km", "mean"),
            median_disagreement_km=("disagreement_km", "median"),
            maximum_disagreement_km=("disagreement_km", "max"),
            short_interval_pairs=("short_interval_flag", "sum"),
            long_interval_pairs=("long_interval_flag", "sum"),
            large_disagreement_pairs=(
                "large_disagreement_flag",
                "sum",
            ),
            extreme_disagreement_pairs=(
                "extreme_disagreement_flag",
                "sum",
            ),
            position_norm_flags=("position_norm_flag", "sum"),
        )
        .reset_index()
    )

    summary["positive_rate_percent"] = (
        100.0 * summary["positive_count"]
        / summary["labeled_pairs"].replace(0, np.nan)
    )

    return summary


def print_pipeline_summary(df, pair_count, reproduction_passed):
    total = len(df)
    labeled = int((df["status"] == "labeled").sum())
    failed = total - labeled

    positives = int(df["label_1km"].fillna(0).sum())
    negatives = int(
        (df["label_1km"].fillna(-1) == 0).sum()
    )

    print("\n" + "=" * 76)
    print("FINAL CONSECUTIVE TLE PIPELINE SUMMARY")
    print("=" * 76)

    print(f"Total consecutive pairs:          {total:,}")
    print(f"Successfully labeled pairs:       {labeled:,}")
    print(f"Pairs without valid labels:       {failed:,}")
    print(f"Negative labels (0):              {negatives:,}")
    print(f"Positive labels (1):              {positives:,}")

    if labeled:
        print(
            f"Positive rate:                    "
            f"{100.0 * positives / labeled:.4f}%"
        )

        print(
            f"Mean disagreement:                "
            f"{df['disagreement_km'].mean():.6f} km"
        )

        print(
            f"Median disagreement:              "
            f"{df['disagreement_km'].median():.6f} km"
        )

        print(
            f"95th percentile disagreement:     "
            f"{df['disagreement_km'].quantile(0.95):.6f} km"
        )

        print(
            f"Maximum disagreement:            "
            f"{df['disagreement_km'].max():.6f} km"
        )

    print("\nQUALITY COUNTS")
    print("-" * 76)

    status_counts = df["status"].value_counts(dropna=False)

    for status, count in status_counts.items():
        print(f"{str(status):<36} {int(count):>10,}")

    print("\nANOMALY FLAGS")
    print("-" * 76)

    print(
        "Intervals under 1 minute:         "
        f"{int(df['short_interval_flag'].sum()):,}"
    )
    print(
        "Intervals over 48 hours:          "
        f"{int(df['long_interval_flag'].sum()):,}"
    )
    print(
        "Disagreement over 10 km:          "
        f"{int(df['large_disagreement_flag'].sum()):,}"
    )
    print(
        "Disagreement over 100 km:         "
        f"{int(df['extreme_disagreement_flag'].sum()):,}"
    )
    print(
        "Position norm outside 6000-8000:  "
        f"{int(df['position_norm_flag'].sum()):,}"
    )

    print("\nREPRODUCTION CHECK")
    print("-" * 76)
    print(
        "Previously reproduced pairs:     "
        f"{'PASS' if reproduction_passed else 'REVIEW REQUIRED'}"
    )

    print("\nINTERVAL DISTRIBUTION")
    print("-" * 76)

    for category, count in (
        df["interval_category"].value_counts().items()
    ):
        print(f"{category:<36} {int(count):>10,}")

    print("\nTOP 15 DISAGREEMENTS")
    print("-" * 76)

    top = (
        df[df["status"] == "labeled"]
        .nlargest(15, "disagreement_km")
    )

    for _, row in top.iterrows():
        print(
            f"Pair {int(row['pair_id']):>7} | "
            f"{str(row['satellite_name']):<20} | "
            f"{row['interval_seconds']:>12.3f} sec | "
            f"{row['disagreement_km']:>14.6f} km | "
            f"label={int(row['label_1km'])}"
        )

    print("\nOUTPUT FILES")
    print("-" * 76)
    print(LABELED_FILE)
    print(SATELLITE_SUMMARY_FILE)
    print(INTERVAL_SUMMARY_FILE)
    print(ANOMALY_FILE)

    print("=" * 76)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 76)
    print("O-RAP: CONSECUTIVE TLE SGP4 LABELING PIPELINE")
    print("=" * 76)

    print(f"Project root: {ROOT}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Label threshold: {LABEL_THRESHOLD_KM} km")
    print("Pairing: immediately consecutive distinct epochs")
    print("Target time: reference TLE epoch")
    print("Frame: TEME")
    print("Interval filtering: NONE")
    print("Large disagreement filtering: NONE")

    pairs = load_and_pair_all_datasets()

    print("\nSTAGE 2: FULL SGP4 PROPAGATION AND LABELING")
    print("-" * 72)

    results = []

    for index, pair in enumerate(pairs, start=1):
        result = process_pair(pair)

        result["interval_category"] = interval_category(
            pair["interval_seconds"]
        )

        results.append(result)

        if index % 10000 == 0 or index == len(pairs):
            labeled_so_far = sum(
                row["status"] == "labeled" for row in results
            )

            print(
                f"Processed {index:,}/{len(pairs):,} | "
                f"labeled={labeled_so_far:,} | "
                f"not_labeled={index - labeled_so_far:,}"
            )

    df = pd.DataFrame(results)

    # Known pair reproduction checks.
    reproduction_passed = verify_known_reproductions(results)

    # Verify every pair is represented exactly once.
    pair_count_passed = (
        len(df) == len(pairs)
        and df["pair_id"].is_unique
    )

    print("\nPAIR UNIQUENESS CHECK:")
    print("PASS" if pair_count_passed else "FAIL")

    if not pair_count_passed:
        raise RuntimeError("Pair uniqueness verification failed.")

    # Save complete master dataset, retaining failed rows.
    df.to_csv(LABELED_FILE, index=False)

    # Satellite-level and interval-level summaries.
    satellite_summary = make_satellite_summary(df)
    interval_summary = make_interval_summary(df)

    satellite_summary.to_csv(
        SATELLITE_SUMMARY_FILE,
        index=False,
    )

    interval_summary.to_csv(
        INTERVAL_SUMMARY_FILE,
        index=False,
    )

    # Keep every large disagreement, extreme disagreement,
    # unusual interval, position norm flag, and failed pair.
    anomaly_mask = (
        df["status"].ne("labeled")
        | df["short_interval_flag"]
        | df["long_interval_flag"]
        | df["large_disagreement_flag"]
        | df["position_norm_flag"]
        | df["known_outlier_pair"]
    )

    anomalies = df[anomaly_mask].copy()

    anomalies.to_csv(ANOMALY_FILE, index=False)

    print_pipeline_summary(
        df,
        len(pairs),
        reproduction_passed,
    )

    print("\nPIPELINE COMPLETED.")
    print(
        "Review the quality summary and anomaly list "
        "before using labels for model training."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nPIPELINE STOPPED")
        print(f"Error: {exc}")
        traceback.print_exc()
        raise
