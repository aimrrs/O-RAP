
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

file_path = (
    ROOT / "data" / "raw" /
    "aqua_27424_gp_history.json"
)

with file_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 65)
print("O-RAP RAW TLE JSON SCHEMA INSPECTION")
print("=" * 65)

print("\nTop-level type:", type(data).__name__)

if isinstance(data, list):
    records = data

elif isinstance(data, dict):
    print("\nTop-level keys:", list(data.keys()))

    records = next(
        (value for value in data.values() if isinstance(value, list)),
        None
    )

    if records is None:
        raise ValueError("Could not locate record list in JSON.")

else:
    raise ValueError("Unexpected JSON structure.")

print("\nRecord count:", len(records))

if not records:
    raise ValueError("No records found.")

print("\nFIRST RECORD FIELD NAMES:")
print(list(records[0].keys()))

print("\nFIRST RECORD CONTENT:")
print(json.dumps(records[0], indent=2))

print("\nSECOND RECORD FIELD NAMES:")
if len(records) > 1:
    print(list(records[1].keys()))

print("\nInspection completed.")
