
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import math

# ============================================================
# O-RAP: LEAKAGE-SAFE CHRONOLOGICAL DATASET SPLITTING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "consecutive_tle_pipeline"
    / "consecutive_tle_labeled_dataset_v2.csv"
)

OUTPUT_DIR = ROOT / "results" / "modeling" / "splits"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
# Remaining 15% is reserved for testing.

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"


def parse_epoch(value):
    """Parse UTC epoch timestamps from the dataset."""
    return datetime.fromisoformat(value.strip())


def read_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    with INPUT_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "pair_id",
            "satellite_key",
            "satellite_name",
            "source_gp_id",
            "reference_gp_id",
            "source_epoch",
            "reference_epoch",
            "interval_hours",
            "disagreement_km",
            "label_1km",
            "status",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        rows = list(reader)
        fieldnames = reader.fieldnames

    return rows, fieldnames


def main():
    print("=" * 72)
    print("O-RAP LEAKAGE-SAFE CHRONOLOGICAL SPLIT")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_dataset()

    print(f"\nInput rows: {len(rows):,}")

    # --------------------------------------------------------
    # 1. Validate and collect unique epochs per satellite
    # --------------------------------------------------------

    epochs_by_satellite = defaultdict(set)
    satellite_names = {}

    for row in rows:
        satellite = row["satellite_key"]

        satellite_names[satellite] = row["satellite_name"]

        source_epoch = parse_epoch(row["source_epoch"])
        reference_epoch = parse_epoch(row["reference_epoch"])

        if reference_epoch <= source_epoch:
            raise ValueError(
                f"Invalid chronological pair: {row['pair_id']}"
            )

        epochs_by_satellite[satellite].add(source_epoch)
        epochs_by_satellite[satellite].add(reference_epoch)

    # --------------------------------------------------------
    # 2. Assign each unique TLE epoch to one partition
    # --------------------------------------------------------

    epoch_partition = {}
    satellite_epoch_counts = {}

    for satellite, epoch_set in epochs_by_satellite.items():

        ordered_epochs = sorted(epoch_set)
        total_epochs = len(ordered_epochs)

        train_end = math.floor(
            total_epochs * TRAIN_RATIO
        )

        validation_end = math.floor(
            total_epochs *
            (TRAIN_RATIO + VALIDATION_RATIO)
        )

        if train_end < 2 or validation_end >= total_epochs:
            raise ValueError(
                f"Insufficient epochs for {satellite}"
            )

        for index, epoch in enumerate(ordered_epochs):

            if index < train_end:
                partition = "train"

            elif index < validation_end:
                partition = "validation"

            else:
                partition = "test"

            epoch_partition[(satellite, epoch)] = partition

        satellite_epoch_counts[satellite] = {
            "total": total_epochs,
            "train": train_end,
            "validation": validation_end - train_end,
            "test": total_epochs - validation_end,
        }

    # --------------------------------------------------------
    # 3. Assign pairs only if both epochs share a partition
    # --------------------------------------------------------

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }

    boundary_rows = []

    for row in rows:

        satellite = row["satellite_key"]

        source_epoch = parse_epoch(row["source_epoch"])
        reference_epoch = parse_epoch(row["reference_epoch"])

        source_partition = epoch_partition[
            (satellite, source_epoch)
        ]

        reference_partition = epoch_partition[
            (satellite, reference_epoch)
        ]

        if source_partition == reference_partition:
            split_rows[source_partition].append(row)

        else:
            row_copy = dict(row)
            row_copy["source_partition"] = source_partition
            row_copy["reference_partition"] = reference_partition
            boundary_rows.append(row_copy)

    # --------------------------------------------------------
    # 4. Save split files
    # --------------------------------------------------------

    for partition, partition_rows in split_rows.items():

        output_file = OUTPUT_DIR / f"{partition}.csv"

        with output_file.open(
            "w", encoding="utf-8", newline=""
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(partition_rows)

        print(
            f"{partition.upper():12s}: "
            f"{len(partition_rows):,} rows"
        )

    # Save excluded boundary-crossing pairs.

    boundary_file = OUTPUT_DIR / "boundary_excluded_pairs.csv"

    boundary_fields = fieldnames + [
        "source_partition",
        "reference_partition",
    ]

    with boundary_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=boundary_fields
        )

        writer.writeheader()
        writer.writerows(boundary_rows)

    # --------------------------------------------------------
    # 5. Verify no TLE epoch is shared between partitions
    # --------------------------------------------------------

    partition_epochs = {
        name: set() for name in split_rows
    }

    partition_gp_ids = {
        name: set() for name in split_rows
    }

    for partition, partition_rows in split_rows.items():

        for row in partition_rows:

            satellite = row["satellite_key"]

            partition_epochs[partition].add(
                (satellite, parse_epoch(row["source_epoch"]))
            )

            partition_epochs[partition].add(
                (satellite, parse_epoch(row["reference_epoch"]))
            )

            partition_gp_ids[partition].add(
                (satellite, row["source_gp_id"])
            )

            partition_gp_ids[partition].add(
                (satellite, row["reference_gp_id"])
            )

    partitions = ["train", "validation", "test"]

    epoch_leakage = False
    gp_id_leakage = False

    print("\nLEAKAGE VERIFICATION")
    print("-" * 72)

    for i, first in enumerate(partitions):

        for second in partitions[i + 1:]:

            shared_epochs = (
                partition_epochs[first]
                & partition_epochs[second]
            )

            shared_gp_ids = (
                partition_gp_ids[first]
                & partition_gp_ids[second]
            )

            if shared_epochs:
                epoch_leakage = True

            if shared_gp_ids:
                gp_id_leakage = True

            print(
                f"{first} vs {second}: "
                f"shared epochs={len(shared_epochs)}, "
                f"shared GP IDs={len(shared_gp_ids)}"
            )

    # --------------------------------------------------------
    # 6. Verify pair IDs are unique across partitions
    # --------------------------------------------------------

    pair_ids = []

    for partition_rows in split_rows.values():
        pair_ids.extend(row["pair_id"] for row in partition_rows)

    pair_id_unique = len(pair_ids) == len(set(pair_ids))

    # --------------------------------------------------------
    # 7. Generate satellite-wise and split summaries
    # --------------------------------------------------------

    summary_file = OUTPUT_DIR / "split_summary.csv"

    summary_fields = [
        "partition",
        "satellite_name",
        "satellite_key",
        "pair_count",
        "positive_count",
        "negative_count",
        "positive_rate_percent",
        "earliest_source_epoch",
        "latest_reference_epoch",
    ]

    summary_rows = []

    for partition, partition_rows in split_rows.items():

        grouped = defaultdict(list)

        for row in partition_rows:
            grouped[row["satellite_key"]].append(row)

        for satellite, satellite_rows in sorted(grouped.items()):

            labels = [
                int(row["label_1km"])
                for row in satellite_rows
            ]

            positives = sum(labels)
            negatives = len(labels) - positives

            summary_rows.append({
                "partition": partition,
                "satellite_name": satellite_names[satellite],
                "satellite_key": satellite,
                "pair_count": len(satellite_rows),
                "positive_count": positives,
                "negative_count": negatives,
                "positive_rate_percent": round(
                    positives / len(labels) * 100, 4
                ),
                "earliest_source_epoch": min(
                    row["source_epoch"] for row in satellite_rows
                ),
                "latest_reference_epoch": max(
                    row["reference_epoch"] for row in satellite_rows
                ),
            })

    with summary_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # 8. Final checks and terminal report
    # --------------------------------------------------------

    total_assigned = sum(
        len(partition_rows)
        for partition_rows in split_rows.values()
    )

    total_positive = sum(
        int(row["label_1km"])
        for partition_rows in split_rows.values()
        for row in partition_rows
    )

    total_negative = total_assigned - total_positive

    print("\nDATASET SUMMARY")
    print("-" * 72)
    print(f"Original pairs:             {len(rows):,}")
    print(f"Assigned pairs:             {total_assigned:,}")
    print(f"Boundary-excluded pairs:    {len(boundary_rows):,}")
    print(f"Positive labels:            {total_positive:,}")
    print(f"Negative labels:            {total_negative:,}")
    print(
        f"Overall positive rate:      "
        f"{total_positive / total_assigned * 100:.4f}%"
    )

    print("\nFINAL VALIDATION")
    print("-" * 72)

    checks = {
        "All rows accounted for": (
            total_assigned + len(boundary_rows) == len(rows)
        ),
        "No shared TLE epochs": not epoch_leakage,
        "No shared GP IDs": not gp_id_leakage,
        "Unique pair IDs": pair_id_unique,
        "At least one row in each split": all(
            len(split_rows[name]) > 0 for name in partitions
        ),
        "No missing labels in assigned data": all(
            row["label_1km"] in {"0", "1"}
            for partition_rows in split_rows.values()
            for row in partition_rows
        ),
    }

    for description, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] {description}"
        )

    print("\nSATELLITE EPOCH COUNTS")
    print("-" * 72)

    for satellite, counts in sorted(satellite_epoch_counts.items()):
        print(
            f"{satellite_names[satellite]:20s} "
            f"total={counts['total']:7,} | "
            f"train={counts['train']:7,} | "
            f"validation={counts['validation']:7,} | "
            f"test={counts['test']:7,}"
        )

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    if not all(checks.values()):
        raise RuntimeError(
            "One or more split validation checks failed."
        )

    print("\nSUCCESS: Leakage-safe splits created and verified.")


if __name__ == "__main__":
    main()
