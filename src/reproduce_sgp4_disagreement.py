
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sgp4.api import Satrec, jday


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SAMPLE_CSV = (
    ROOT / "results" / "consecutive_sgp4_validation"
    / "validation_sample.csv"
)

VALIDATION_CSV = (
    ROOT / "results" / "consecutive_sgp4_validation"
    / "sgp4_validation_results.csv"
)

PAIR_IDS = [207869, 103442]


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


def parse_utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def propagate(tle_line1, tle_line2, target_time):
    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    jd, fr = jday(
        target_time.year,
        target_time.month,
        target_time.day,
        target_time.hour,
        target_time.minute,
        target_time.second
        + target_time.microsecond / 1_000_000.0,
    )

    error, position, velocity = sat.sgp4(jd, fr)

    return {
        "satellite": sat,
        "error": error,
        "position": position,
        "velocity": velocity,
    }


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def main():
    sample = pd.read_csv(SAMPLE_CSV)
    validation = pd.read_csv(VALIDATION_CSV)

    sample["pair_id"] = sample["pair_id"].astype(int)
    validation["pair_id"] = validation["pair_id"].astype(int)

    raw_cache = {}

    print("=" * 76)
    print("INDEPENDENT SGP4 DISAGREEMENT REPRODUCTION")
    print("=" * 76)
    print("Method: Propagate source TLE and reference TLE to reference epoch.")
    print("Frame: TEME (as returned by SGP4)")
    print("Distance unit: kilometres")
    print()

    for pair_id in PAIR_IDS:
        matches = sample[sample["pair_id"] == pair_id]

        if matches.empty:
            print(f"ERROR: Pair {pair_id} not found in validation sample.")
            continue

        pair = matches.iloc[0]
        raw_file = pair["raw_file"]

        if raw_file not in raw_cache:
            raw_path = RAW_DIR / raw_file
            raw_cache[raw_file] = load_json_records(raw_path)

        records = raw_cache[raw_file]

        source_index = int(pair["source_record_index"])
        reference_index = int(pair["reference_record_index"])

        source_record = records[source_index]
        reference_record = records[reference_index]

        source_line1 = source_record["TLE_LINE1"].rstrip()
        source_line2 = source_record["TLE_LINE2"].rstrip()

        reference_line1 = reference_record["TLE_LINE1"].rstrip()
        reference_line2 = reference_record["TLE_LINE2"].rstrip()

        source_epoch = parse_utc(source_record["EPOCH"])
        reference_epoch = parse_utc(reference_record["EPOCH"])

        interval_seconds = (
            reference_epoch - source_epoch
        ).total_seconds()

        # Both TLEs are propagated to exactly the same UTC instant.
        target_time = reference_epoch

        source_result = propagate(
            source_line1,
            source_line2,
            target_time,
        )

        reference_result = propagate(
            reference_line1,
            reference_line2,
            target_time,
        )

        print("-" * 76)
        print(f"PAIR ID: {pair_id}")
        print(f"Satellite: {pair['satellite_name']}")
        print(f"NORAD ID: {pair['norad_id']}")
        print(f"Source GP ID: {source_record.get('GP_ID')}")
        print(f"Reference GP ID: {reference_record.get('GP_ID')}")
        print(f"Source epoch: {source_epoch.isoformat()}")
        print(f"Reference epoch: {reference_epoch.isoformat()}")
        print(f"Interval: {interval_seconds:.9f} seconds")
        print(f"Target propagation time: {target_time.isoformat()}")

        print("\nSOURCE TLE:")
        print(source_line1)
        print(source_line2)

        print("\nREFERENCE TLE:")
        print(reference_line1)
        print(reference_line2)

        print("\nSGP4 PROPAGATION STATUS:")
        print(f"Source SGP4 error code: {source_result['error']}")
        print(f"Reference SGP4 error code: {reference_result['error']}")

        if source_result["error"] != 0 or reference_result["error"] != 0:
            print("RESULT: Propagation error. Disagreement not evaluated.")
            continue

        source_position = source_result["position"]
        reference_position = reference_result["position"]

        source_velocity = source_result["velocity"]
        reference_velocity = reference_result["velocity"]

        difference = [
            source_position[i] - reference_position[i]
            for i in range(3)
        ]

        disagreement = norm(difference)

        print("\nSOURCE POSITION (TEME, km):")
        print(
            f"X = {source_position[0]:.9f}, "
            f"Y = {source_position[1]:.9f}, "
            f"Z = {source_position[2]:.9f}"
        )

        print("\nREFERENCE POSITION (TEME, km):")
        print(
            f"X = {reference_position[0]:.9f}, "
            f"Y = {reference_position[1]:.9f}, "
            f"Z = {reference_position[2]:.9f}"
        )

        print("\nPOSITION DIFFERENCE (SOURCE - REFERENCE, km):")
        print(
            f"dX = {difference[0]:.9f}, "
            f"dY = {difference[1]:.9f}, "
            f"dZ = {difference[2]:.9f}"
        )

        print("\nPOSITION NORMS:")
        print(f"Source norm: {norm(source_position):.9f} km")
        print(f"Reference norm: {norm(reference_position):.9f} km")

        print("\nVELOCITY NORMS:")
        print(f"Source: {norm(source_velocity):.9f} km/s")
        print(f"Reference: {norm(reference_velocity):.9f} km/s")

        print("\nINDEPENDENT 3D DISAGREEMENT:")
        print(f"{disagreement:.12f} km")

        print(f"Label at 1 km threshold: {int(disagreement > 1.0)}")

        previous = validation[
            validation["pair_id"] == pair_id
        ]

        if not previous.empty:
            reported = float(previous.iloc[0]["disagreement_km"])
            difference_from_report = abs(disagreement - reported)

            print("\nCOMPARISON WITH PREVIOUS VALIDATION:")
            print(f"Previously reported: {reported:.12f} km")
            print(f"Independently calculated: {disagreement:.12f} km")
            print(
                f"Absolute difference: "
                f"{difference_from_report:.12f} km"
            )

            if difference_from_report < 0.000001:
                print("REPRODUCTION: MATCH")
            else:
                print("REPRODUCTION: MISMATCH")

    print()
    print("=" * 76)
    print("END OF INDEPENDENT REPRODUCTION")
    print("=" * 76)


if __name__ == "__main__":
    main()
