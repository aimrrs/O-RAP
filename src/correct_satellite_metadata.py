
import csv
from collections import Counter

INPUT = "results/consecutive_tle_pipeline/consecutive_tle_labeled_dataset.csv"
OUTPUT = "results/consecutive_tle_pipeline/consecutive_tle_labeled_dataset_v2.csv"

# Verified using the NORAD IDs and corresponding raw dataset filenames.
SATELLITE_NAMES = {
    "25544": "ISS (ZARYA)",
    "25994": "TERRA",
    "27424": "AQUA",
    "28054": "DMSP 5D-3 F16 (USA 172)",
    "28376": "AURA",
    "29499": "METOP-A",
    "31598": "COSMO-SKYMED 1",
    "31698": "TERRASAR-X",
    "32382": "RADARSAT-2",
    "33591": "NOAA-19",
    "37849": "SUOMI NPP",
    "39084": "LANDSAT 8",
    "40697": "SENTINEL-2A",
    "43013": "NOAA-20",
}


def main():
    original_counts = Counter()
    corrected_counts = Counter()
    label_counts_before = Counter()
    label_counts_after = Counter()

    total = 0

    with open(INPUT, "r", encoding="utf-8-sig", newline="") as source, \
         open(OUTPUT, "w", encoding="utf-8", newline="") as destination:

        reader = csv.DictReader(source)

        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header.")

        required = {"norad_id", "satellite_name", "label_1km"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(f"Missing required columns: {sorted(missing)}")

        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames
        )
        writer.writeheader()

        for row in reader:
            norad_id = row["norad_id"].strip()

            if norad_id not in SATELLITE_NAMES:
                raise RuntimeError(
                    f"Unknown NORAD ID at row {total + 2}: {norad_id}"
                )

            original_counts[
                (norad_id, row["satellite_name"])
            ] += 1

            label_counts_before[row["label_1km"]] += 1

            # Change only the satellite_name field.
            row["satellite_name"] = SATELLITE_NAMES[norad_id]

            corrected_counts[
                (norad_id, row["satellite_name"])
            ] += 1

            label_counts_after[row["label_1km"]] += 1

            writer.writerow(row)
            total += 1

            if total % 50000 == 0:
                print(f"Processed {total:,} rows...")

    print("\n" + "=" * 65)
    print("SATELLITE METADATA CORRECTION")
    print("=" * 65)

    print(f"Total rows processed: {total:,}")
    print(f"Original dataset: {INPUT}")
    print(f"Corrected dataset: {OUTPUT}")

    print("\nCORRECTED SATELLITE COUNTS")
    for (norad, name), count in sorted(corrected_counts.items()):
        print(f"{norad:>6} | {name:<28} | {count:>8,}")

    print("\nVERIFICATION")
    print("Total rows preserved:", sum(corrected_counts.values()))
    print("Original label counts:", dict(label_counts_before))
    print("Corrected label counts:", dict(label_counts_after))

    if label_counts_before != label_counts_after:
        raise RuntimeError("Label counts changed unexpectedly.")

    if sum(corrected_counts.values()) != total:
        raise RuntimeError("Row count verification failed.")

    if len({norad for norad, _ in corrected_counts}) != 14:
        raise RuntimeError("Expected exactly 14 satellites.")

    print("\nPASS: All rows and labels preserved.")
    print("PASS: Satellite names mapped using NORAD IDs.")
    print("PASS: Original dataset left unchanged.")


if __name__ == "__main__":
    main()
