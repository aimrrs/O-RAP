from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = ROOT / "results" / "modeling"

TARGET_FILES = {
    "model_comparison.csv",
    "orbital_only_calibration_metrics.csv",
    "loso_by_satellite.csv",
    "loso_pooled_metrics.json",
    "threshold_sensitivity_summary.csv",
    "split_summary.csv",
}

print("=" * 75)
print("O-RAP: FINAL RESULTS SOURCE INSPECTION")
print("=" * 75)

found = {}

for path in SEARCH_DIR.rglob("*"):
    if path.is_file() and path.name in TARGET_FILES:
        found[path.name] = path

for filename in sorted(TARGET_FILES):
    print(f"\n{'-' * 75}")
    print(f"FILE: {filename}")
    print("-" * 75)

    path = found.get(filename)

    if path is None:
        print("[NOT FOUND]")
        continue

    print(f"PATH: {path.relative_to(ROOT)}")

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            print("COLUMNS:")
            print(reader.fieldnames)

            first_rows = []
            for row in reader:
                first_rows.append(row)
                if len(first_rows) == 2:
                    break

            print("SAMPLE ROWS:")
            for row in first_rows:
                print(row)

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(data, indent=2))

print("\n" + "=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)