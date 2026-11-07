"""Small, dependency-light web interface for the O-RAP research model."""

from __future__ import annotations

import csv
import json
import math
import mimetypes
import re
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "results" / "modeling" / "experiments" / "models"
ORBITAL_RESULTS = ROOT / "results" / "modeling" / "experiments" / "orbital_only_analysis"

# Keep this order synchronized with src/analyze_orbital_only_orap.py.
FEATURES = [
    "mean_motion", "eccentricity", "inclination", "raan",
    "argument_of_pericenter", "mean_anomaly", "bstar",
    "mean_motion_dot", "mean_motion_ddot", "semimajor_axis",
    "orbital_period", "apoapsis", "periapsis",
]
RECORD_FIELDS = {
    "mean_motion": "MEAN_MOTION", "eccentricity": "ECCENTRICITY",
    "inclination": "INCLINATION", "raan": "RA_OF_ASC_NODE",
    "argument_of_pericenter": "ARG_OF_PERICENTER",
    "mean_anomaly": "MEAN_ANOMALY", "bstar": "BSTAR",
    "mean_motion_dot": "MEAN_MOTION_DOT",
    "mean_motion_ddot": "MEAN_MOTION_DDOT",
    "semimajor_axis": "SEMIMAJOR_AXIS", "orbital_period": "PERIOD",
    "apoapsis": "APOAPSIS", "periapsis": "PERIAPSIS",
}
RAW_FILES = {
    "AQUA": "aqua_27424_gp_history.json",
    "AURA": "aura_28376_gp_history.json",
    "COSMO-SKYMED 1": "cosmo_skymed_1_31598_gp_history.json",
    "DMSP 5D-3 F16 (USA 172)": "dmsp_5d_3_f16_28054_gp_history.json",
    "ISS (ZARYA)": "iss_25544_gp_history.json",
    "LANDSAT 8": "landsat_8_39084_gp_history.json",
    "METOP-A": "metop_a_29499_gp_history.json",
    "NOAA-19": "noaa_19_33591_gp_history.json",
    "NOAA-20": "noaa_20_43013_gp_history.json",
    "RADARSAT-2": "radarsat_2_32382_gp_history.json",
    "SENTINEL-2A": "sentinel_2a_40697_gp_history.json",
    "SUOMI NPP": "suomi_npp_37849_gp_history.json",
    "TERRA": "terra_25994_gp_history.json",
    "TERRASAR-X": "terrasar_x_31698_gp_history.json",
}


@lru_cache(maxsize=1)
def load_runtime():
    """Load the committed estimator and calibrator only when needed."""
    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies first (python -m pip install -r requirements.txt).") from exc
    model_path = MODEL_DIR / "orbital_only_xgboost.joblib"
    calibrator_path = ORBITAL_RESULTS / "orbital_only_platt_calibrator.joblib"
    if not model_path.is_file() or not calibrator_path.is_file():
        raise RuntimeError("The committed orbital-only model or its calibrator is missing.")
    return joblib.load(model_path), joblib.load(calibrator_path), np


def records_for_satellite(name: str) -> list[dict]:
    filename = RAW_FILES.get(name)
    if filename is None:
        raise ValueError("Choose one of the satellites in the repository data.")
    with (ROOT / "data" / "raw" / filename).open(encoding="utf-8") as stream:
        records = json.load(stream)
    return records


def record_features(record: dict) -> dict:
    try:
        features = {key: float(record[field]) for key, field in RECORD_FIELDS.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("This TLE record is missing a required orbital field.") from exc
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("This TLE record contains a non-finite orbital field.")
    return features


def satellite_options() -> list[dict]:
    options = []
    for name, filename in RAW_FILES.items():
        with (ROOT / "data" / "raw" / filename).open(encoding="utf-8") as stream:
            records = json.load(stream)
        valid = [(index, record) for index, record in enumerate(records)
                 if all(field in record for field in RECORD_FIELDS.values())]
        if not valid:
            continue
        index, record = valid[-1]
        options.append({
            "name": name,
            "record_index": index,
            "epoch": record.get("EPOCH", ""),
            "features": record_features(record),
            "tle": record_to_tle(record),
        })
    return options


def record_to_tle(record: dict) -> str:
    """Return the original TLE lines stored in the source GP record."""
    try:
        lines = (str(record["TLE_LINE1"]).strip(), str(record["TLE_LINE2"]).strip())
        if not lines[0].startswith("1 ") or not lines[1].startswith("2 "):
            raise ValueError
        return "\n".join(lines)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("This project record cannot be formatted as a standard TLE set.") from exc


def _tle_checksum(line: str) -> str:
    return str(sum(int(char) for char in line if char.isdigit()) + line.count("-"))[-1]


def parse_tle(tle_text: str) -> dict:
    """Parse a two-line TLE, deriving the same 13 orbital fields used by O-RAP."""
    try:
        lines = [line.strip() for line in tle_text.splitlines() if line.strip()]
        if len(lines) == 3 and not lines[0].startswith("1 "):
            lines = lines[1:]
        if len(lines) != 2 or not lines[0].startswith("1 ") or not lines[1].startswith("2 "):
            raise ValueError("Paste exactly two TLE lines (line 1 and line 2).")
        if len(lines[0]) < 68 or len(lines[1]) < 68:
            raise ValueError("Each TLE line must contain at least 68 characters.")
        if lines[0][2:7] != lines[1][2:7]:
            raise ValueError("The satellite catalogue numbers on the two TLE lines do not match.")
        mean_motion = float(lines[1][52:63])
        eccentricity = float(f"0.{lines[1][26:33].strip()}")
        if mean_motion <= 0 or not 0 <= eccentricity < 1:
            raise ValueError("The TLE contains invalid mean motion or eccentricity.")
        inclination = float(lines[1][8:16])
        raan = float(lines[1][17:25])
        argument = float(lines[1][34:42])
        anomaly = float(lines[1][43:51])
        ndot_text = lines[0][33:43].strip()
        mean_motion_dot = float(ndot_text.replace("-.", "-0.").replace("+.", "0.").replace(".", "0.", 1) if ndot_text.startswith((".", "+.", "-.")) else ndot_text)
        def implied_value(field: str) -> float:
            field = field.strip()
            if not field:
                raise ValueError
            if field[0] in "+-":
                sign = -1 if field[0] == "-" else 1
                mantissa_text, exponent_text = field[1:6], field[6:8]
            else:
                sign = 1
                mantissa_text, exponent_text = field[:5], field[5:7]
            mantissa = sign * int(mantissa_text)
            exponent = int(exponent_text)
            return mantissa * 1e-5 * (10 ** exponent)
        mean_motion_ddot = implied_value(lines[0][44:52])
        bstar = implied_value(lines[0][53:61])
        period = 1440.0 / mean_motion
        semimajor = (398600.4418 / ((mean_motion * 2 * math.pi / 86400) ** 2)) ** (1 / 3)
        earth_radius = 6378.137
        return {
            "mean_motion": mean_motion, "eccentricity": eccentricity, "inclination": inclination,
            "raan": raan, "argument_of_pericenter": argument, "mean_anomaly": anomaly,
            "bstar": bstar, "mean_motion_dot": mean_motion_dot, "mean_motion_ddot": mean_motion_ddot,
            "semimajor_axis": semimajor, "orbital_period": period,
            "apoapsis": semimajor * (1 + eccentricity) - earth_radius,
            "periapsis": semimajor * (1 - eccentricity) - earth_radius,
        }
    except (IndexError, OverflowError, ValueError) as exc:
        raise ValueError("The TLE lines are incomplete or contain invalid values.") from exc


def predict(features: dict) -> dict:
    if set(features) != set(FEATURES):
        raise ValueError("Provide all 13 orbital values.")
    values = []
    for key in FEATURES:
        try:
            value = float(features[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a finite number.")
        values.append(value)

    model, calibrator, np = load_runtime()
    raw = float(model.predict_proba(np.asarray([values], dtype=np.float64))[0, 1])
    clipped = min(max(raw, 1e-7), 1 - 1e-7)
    logit = math.log(clipped / (1 - clipped))
    calibrated = float(calibrator.predict_proba(np.asarray([[logit]], dtype=np.float64))[0, 1])
    config = json.loads((ORBITAL_RESULTS / "orbital_only_analysis_configuration.json").read_text(encoding="utf-8"))
    threshold = float(config["validation_selected_f1_threshold"])
    return {"probability": calibrated, "raw_probability": raw, "threshold": threshold,
            "flagged": calibrated >= threshold, "model": "orbital_only_xgboost + Platt scaling",
            "model_details": {"estimator": "XGBoost classifier (orbital-only)", "calibration": "Platt scaling",
                              "input_features": len(FEATURES), "feature_names": FEATURES,
                              "decision_threshold": threshold, "threshold_selection": "F1-maximizing threshold on validation data",
                              "training_data": "242,547 consecutive TLE pairs; chronological 70/15/15 split",
                              "label": "more than 1 km disagreement between two SGP4 positions at the newer TLE epoch"}}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def research_data() -> dict:
    config = json.loads((ORBITAL_RESULTS / "orbital_only_analysis_configuration.json").read_text(encoding="utf-8"))
    calibrated = config["test_metrics_calibrated"]
    model_rows = read_csv(ROOT / "results" / "final_tables" / "table_02_model_comparison.csv")
    orbital_rows = read_csv(ORBITAL_RESULTS / "orbital_only_per_satellite.csv")
    dataset_summary = read_csv(ROOT / "results" / "final_tables" / "table_01_dataset_split.csv")
    split_summary = read_csv(ROOT / "results" / "final_tables" / "table_01_dataset_split.csv")
    loso = json.loads((ROOT / "results" / "modeling" / "experiments" / "loso" / "loso_pooled_metrics.json").read_text(encoding="utf-8"))
    thresholds = read_csv(ROOT / "results" / "final_tables" / "table_06_threshold_sensitivity.csv")
    calibration = read_csv(ORBITAL_RESULTS / "orbital_only_calibration_metrics.csv")
    retained_pairs = sum(int(row["Samples"]) for row in dataset_summary if row["Partition"] != "Boundary-excluded pairs")
    boundary_excluded = next((int(row["Samples"]) for row in dataset_summary if row["Partition"] == "Boundary-excluded pairs"), 0)
    return {
        "test": {"samples": calibrated["samples"], "positives": int(calibrated["positives"]),
                 "prevalence": calibrated["positive_rate_percent"], "positive_rate_percent": calibrated["positive_rate_percent"], "pr_auc": calibrated["pr_auc"],
                 "roc_auc": calibrated["roc_auc"], "f1": calibrated["f1"],
                 "brier": calibrated["brier_score"], "brier_score": calibrated["brier_score"], "log_loss": calibrated["log_loss"],
                 "precision": calibrated["precision"], "recall": calibrated["recall"],
                 "threshold": config["validation_selected_f1_threshold"]},
        "models": model_rows,
        "satellites": orbital_rows,
        "split_table": dataset_summary,
        "dataset": {"satellites": len(RAW_FILES), "pairs": retained_pairs, "boundary_excluded": boundary_excluded, "features": len(FEATURES)},
        "splits": split_summary,
        "loso": loso,
        "thresholds": thresholds,
        "calibration": calibration,
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The browser may navigate away while a data response is in flight.
            pass

    def do_GET(self):  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/satellites":
                return self.send_json({"satellites": satellite_options()})
            if route == "/api/research":
                return self.send_json(research_data())
            if route == "/api/health":
                return self.send_json({"status": "ok", "model_files_present":
                                       (MODEL_DIR / "orbital_only_xgboost.joblib").is_file() and
                                       (ORBITAL_RESULTS / "orbital_only_platt_calibrator.joblib").is_file()})
            if route in ("/", "/research", "/about") or route in ("/index.html", "/styles.css", "/app.js"):
                asset = "index.html" if route in ("/", "/research", "/about") else route.lstrip("/")
                target = (WEB_ROOT / asset).resolve()
                if target.parent != WEB_ROOT.resolve() or not target.is_file():
                    return self.send_json({"error": "Not found."}, 404)
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_json({"error": "Not found."}, 404)
        except Exception as exc:  # send a simple, useful API error
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/predict":
            return self.send_json({"error": "Not found."}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 16_384:
                return self.send_json({"error": "Request is too large."}, 413)
            payload = json.loads(self.rfile.read(size))
            result = predict(parse_tle(payload.get("tle", "")))
            self.send_json(result)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"O-RAP is running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping O-RAP.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
