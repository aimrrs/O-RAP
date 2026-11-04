
import json
import math
from pathlib import Path

import pandas as pd
from sgp4.api import Satrec


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_CSV = ROOT / "results" / "consecutive_sgp4_validation" / "validation_sample.csv"
RESULTS_CSV = ROOT / "results" / "consecutive_sgp4_validation" / "sgp4_validation_results.csv"

OUT_DIR = ROOT / "results" / "tle_forensic_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# High-disagreement cases and unusually short-interval cases.
FOCUS_PAIR_IDS = [
    206770, 81943, 210424, 4897, 53018,
    67695, 83630, 80037, 103442, 103443,
    228823, 207869,
]


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


def checksum_ok(line):
    if not isinstance(line, str) or len(line) != 69:
        return False

    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1

    return str(total % 10) == line[68]


def angular_difference_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def extract_tle_details(record, role, pair_id, record_index):
    line1 = str(record.get("TLE_LINE1", "")).rstrip()
    line2 = str(record.get("TLE_LINE2", "")).rstrip()

    result = {
        "pair_id": pair_id,
        "role": role,
        "record_index": record_index,
        "gp_id": record.get("GP_ID"),
        "norad_id_json": record.get("NORAD_CAT_ID"),
        "object_name": record.get("OBJECT_NAME"),
        "epoch_json": record.get("EPOCH"),
        "creation_date": record.get("CREATION_DATE"),
        "line1": line1,
        "line2": line2,
        "line1_length": len(line1),
        "line2_length": len(line2),
        "line1_checksum_ok": checksum_ok(line1),
        "line2_checksum_ok": checksum_ok(line2),
        "line1_satellite_number": line1[2:7].strip() if len(line1) >= 7 else "",
        "line2_satellite_number": line2[2:7].strip() if len(line2) >= 7 else "",
    }

    try:
        sat = Satrec.twoline2rv(line1, line2)

        result.update({
            "tle_inclination_deg": math.degrees(sat.inclo),
            "tle_raan_deg": math.degrees(sat.nodeo),
            "tle_eccentricity": sat.ecco,
            "tle_argument_perigee_deg": math.degrees(sat.argpo),
            "tle_mean_anomaly_deg": math.degrees(sat.mo),
            "tle_mean_motion_rev_day": sat.no_kozai * 1440.0 / (2.0 * math.pi),
            "tle_bstar": sat.bstar,
            "tle_epoch_year": sat.epochyr,
            "tle_epoch_day": sat.epochdays,
            "sgp4_parse_ok": True,
        })
    except Exception as exc:
        result["sgp4_parse_ok"] = False
        result["sgp4_parse_error"] = str(exc)

    return result


def main():
    sample = pd.read_csv(SAMPLE_CSV)
    validation = pd.read_csv(RESULTS_CSV)

    sample["pair_id"] = sample["pair_id"].astype(int)
    validation["pair_id"] = validation["pair_id"].astype(int)

    selected = sample[sample["pair_id"].isin(FOCUS_PAIR_IDS)].copy()
    selected = selected.merge(
        validation[["pair_id", "disagreement_km", "unreliable_1km"]],
        on="pair_id",
        how="left",
    )

    if selected.empty:
        raise RuntimeError("None of the requested pair IDs were found.")

    cache = {}
    output_rows = []

    for _, pair in selected.iterrows():
        raw_filename = pair["raw_file"]

        if raw_filename not in cache:
            raw_path = RAW_DIR / raw_filename
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            cache[raw_filename] = load_json_records(raw_path)

        records = cache[raw_filename]

        for role, index_column in (
            ("source", "source_record_index"),
            ("reference", "reference_record_index"),
        ):
            index = int(pair[index_column])

            if index < 0 or index >= len(records):
                raise IndexError(
                    f"Pair {pair['pair_id']}: record index {index} "
                    f"outside {raw_filename} ({len(records)} records)"
                )

            record = records[index]

            output_rows.append(
                extract_tle_details(
                    record=record,
                    role=role,
                    pair_id=int(pair["pair_id"]),
                    record_index=index,
                )
            )

    details = pd.DataFrame(output_rows)
    details.to_csv(OUT_DIR / "suspicious_tle_records.csv", index=False)

    # Produce a pair-level comparison of the parsed orbital elements.
    comparison_rows = []

    for pair_id, group in details.groupby("pair_id"):
        source = group[group["role"] == "source"].iloc[0]
        reference = group[group["role"] == "reference"].iloc[0]

        row = {
            "pair_id": pair_id,
            "satellite": selected.loc[
                selected["pair_id"] == pair_id, "satellite_name"
            ].iloc[0],
            "interval_seconds": (
                selected.loc[
                    selected["pair_id"] == pair_id, "epoch_interval_hours"
                ].iloc[0] * 3600.0
            ),
            "disagreement_km": selected.loc[
                selected["pair_id"] == pair_id, "disagreement_km"
            ].iloc[0],
            "source_gp_id": source["gp_id"],
            "reference_gp_id": reference["gp_id"],
            "source_epoch": source["epoch_json"],
            "reference_epoch": reference["epoch_json"],
            "source_creation_date": source["creation_date"],
            "reference_creation_date": reference["creation_date"],
            "source_line1_checksum_ok": source["line1_checksum_ok"],
            "source_line2_checksum_ok": source["line2_checksum_ok"],
            "reference_line1_checksum_ok": reference["line1_checksum_ok"],
            "reference_line2_checksum_ok": reference["line2_checksum_ok"],
            "source_parse_ok": source["sgp4_parse_ok"],
            "reference_parse_ok": reference["sgp4_parse_ok"],
        }

        numeric_fields = [
            "tle_inclination_deg",
            "tle_raan_deg",
            "tle_eccentricity",
            "tle_argument_perigee_deg",
            "tle_mean_anomaly_deg",
            "tle_mean_motion_rev_day",
            "tle_bstar",
        ]

        for field in numeric_fields:
            a = source.get(field)
            b = reference.get(field)

            try:
                a = float(a)
                b = float(b)
                difference = b - a

                if field.endswith("_deg") and field != "tle_inclination_deg":
                    difference = angular_difference_deg(b, a)

                row[f"source_{field}"] = a
                row[f"reference_{field}"] = b
                row[f"delta_{field}"] = difference
            except (TypeError, ValueError):
                row[f"source_{field}"] = a
                row[f"reference_{field}"] = b
                row[f"delta_{field}"] = None

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "pair_comparison.csv", index=False)

    report_path = OUT_DIR / "forensic_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("CONSECUTIVE TLE FORENSIC AUDIT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Pairs inspected: {len(selected)}\n")
        f.write(f"TLE records inspected: {len(details)}\n\n")

        for _, row in comparison.sort_values(
            "disagreement_km", ascending=False
        ).iterrows():
            f.write("-" * 70 + "\n")
            f.write(f"Pair ID: {row['pair_id']}\n")
            f.write(f"Satellite: {row['satellite']}\n")
            f.write(f"Interval (seconds): {row['interval_seconds']}\n")
            f.write(f"Disagreement (km): {row['disagreement_km']}\n")
            f.write(
                f"GP IDs: {row['source_gp_id']} -> "
                f"{row['reference_gp_id']}\n"
            )
            f.write(
                f"Epochs: {row['source_epoch']} -> "
                f"{row['reference_epoch']}\n"
            )
            f.write(
                f"Creation dates: {row['source_creation_date']} -> "
                f"{row['reference_creation_date']}\n"
            )

            for field in numeric_fields:
                f.write(
                    f"{field}: source={row.get('source_' + field)}, "
                    f"reference={row.get('reference_' + field)}, "
                    f"delta={row.get('delta_' + field)}\n"
                )

            f.write(
                "Checksums: "
                f"{row['source_line1_checksum_ok']}/"
                f"{row['source_line2_checksum_ok']} -> "
                f"{row['reference_line1_checksum_ok']}/"
                f"{row['reference_line2_checksum_ok']}\n"
            )
            f.write(
                f"SGP4 parsing: {row['source_parse_ok']} -> "
                f"{row['reference_parse_ok']}\n"
            )

            for role in ("source", "reference"):
                record = details[
                    (details["pair_id"] == row["pair_id"])
                    & (details["role"] == role)
                ].iloc[0]
                f.write(f"\n{role.upper()} TLE:\n")
                f.write(f"{record['line1']}\n")
                f.write(f"{record['line2']}\n")

            f.write("\n")

    print("Forensic audit completed.")
    print(f"Pairs inspected: {len(selected)}")
    print(f"Records inspected: {len(details)}")
    print(f"Output folder: {OUT_DIR}")
    print("Files:")
    print("  forensic_report.txt")
    print("  suspicious_tle_records.csv")
    print("  pair_comparison.csv")


if __name__ == "__main__":
    main()
