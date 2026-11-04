
import csv
from collections import Counter, defaultdict

INPUT = (
    "results/consecutive_tle_pipeline/"
    "consecutive_tle_labeled_dataset_v2.csv"
)

THRESHOLD_KM = 100.0


def main():
    extreme_pairs = []
    satellite_counts = Counter()
    interval_counts = Counter()
    total_rows = 0
    total_positive = 0

    with open(INPUT, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "pair_id", "satellite_name", "norad_id",
            "source_gp_id", "reference_gp_id",
            "source_epoch", "reference_epoch",
            "interval_hours", "disagreement_km",
            "label_1km", "source_creation_date",
            "reference_creation_date"
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(f"Missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1

            if row["label_1km"] == "1":
                total_positive += 1

            disagreement = float(row["disagreement_km"])

            if disagreement > THRESHOLD_KM:
                extreme_pairs.append(row)
                satellite_counts[row["satellite_name"]] += 1
                interval_counts[row["interval_category"]] += 1

    extreme_pairs.sort(
        key=lambda row: float(row["disagreement_km"]),
        reverse=True
    )

    print("\n" + "=" * 90)
    print("EXTREME CONSECUTIVE-TLE DISAGREEMENT AUDIT")
    print("=" * 90)

    print(f"Total dataset pairs: {total_rows:,}")
    print(f"Total positive labels: {total_positive:,}")
    print(f"Threshold: disagreement > {THRESHOLD_KM} km")
    print(f"Extreme pairs: {len(extreme_pairs):,}")
    print(
        f"Extreme pairs as percentage of dataset: "
        f"{len(extreme_pairs) / total_rows * 100:.4f}%"
    )

    print("\nEXTREME PAIRS BY SATELLITE")
    print("-" * 65)

    for satellite, count in satellite_counts.most_common():
        print(f"{satellite:<30} {count:>8,}")

    print("\nEXTREME PAIRS BY INTERVAL CATEGORY")
    print("-" * 65)

    for category, count in interval_counts.most_common():
        print(f"{category:<30} {count:>8,}")

    print("\nALL EXTREME PAIRS, SORTED BY DISAGREEMENT")
    print("-" * 110)

    for row in extreme_pairs:
        print(
            f"\nPair ID: {row['pair_id']}"
            f"\nSatellite: {row['satellite_name']} "
            f"(NORAD {row['norad_id']})"
            f"\nDisagreement: {float(row['disagreement_km']):,.6f} km"
            f"\nInterval: {float(row['interval_hours']):.9f} hours"
            f"\nSource GP ID: {row['source_gp_id']}"
            f"\nReference GP ID: {row['reference_gp_id']}"
            f"\nSource epoch: {row['source_epoch']}"
            f"\nReference epoch: {row['reference_epoch']}"
            f"\nSource creation: {row['source_creation_date']}"
            f"\nReference creation: {row['reference_creation_date']}"
            f"\nRTN differences (km): "
            f"R={row.get('radial_difference_km', '')}, "
            f"T={row.get('along_track_difference_km', '')}, "
            f"N={row.get('cross_track_difference_km', '')}"
            f"\nInterval category: {row['interval_category']}"
            f"\nLabel: {row['label_1km']}"
            f"\nFlags: large={row.get('large_disagreement_flag', '')}, "
            f"extreme={row.get('extreme_disagreement_flag', '')}, "
            f"short={row.get('short_interval_flag', '')}, "
            f"long={row.get('long_interval_flag', '')}"
        )

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE — NO DATA WAS MODIFIED")
    print("=" * 90)


if __name__ == "__main__":
    main()
