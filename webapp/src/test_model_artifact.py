import json
from pathlib import Path

from predict import load_artifacts


def main():
    model, metadata = load_artifacts()
    assert model is not None
    assert metadata["features"]
    calibration_path = Path(__file__).resolve().parents[1] / "models" / "orap_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration.get("method") == "platt_scaling"
    print("Bundled model metadata and retained calibration provenance load successfully.")
    print("The demo inference intentionally does not apply this calibrator.")


if __name__ == "__main__":
    main()
