from pathlib import Path
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "train_temporal.csv"
VAL_FILE = DATA_DIR / "validation_temporal.csv"

MODEL_FILE = MODEL_DIR / "orap_xgboost.json"
CALIBRATION_FILE = MODEL_DIR / "orap_calibration.json"
METADATA_FILE = MODEL_DIR / "orap_metadata.json"


FEATURES = [
    "horizon_hours",
    "source_tle_age_hours",
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "period",
    "apoapsis",
    "periapsis",
]

TARGET = "unreliable_1km"


def probability_to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_val = validation[FEATURES]
    y_val = validation[TARGET].astype(int)

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    scale_pos_weight = negative / positive

    print("O-RAP PRODUCTION MODEL TRAINING")
    print("=" * 40)
    print(f"Training samples:   {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Training positives:  {positive}")
    print(f"Training negatives:  {negative}")
    print(f"Positive rate:       {y_train.mean():.4f}")
    print(f"scale_pos_weight:    {scale_pos_weight:.4f}")

    # Frozen final research configuration.
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    # Validation-only Platt calibration.
    raw_val = model.predict_proba(X_val)[:, 1]
    val_logits = probability_to_logit(raw_val)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(val_logits.reshape(-1, 1), y_val)

    print("\nCalibration:")
    print(f"Coefficient: {calibrator.coef_[0][0]:.6f}")
    print(f"Intercept:   {calibrator.intercept_[0]:.6f}")

    # Save model.
    model.save_model(MODEL_FILE)

    # Save calibration parameters.
    calibration = {
        "method": "platt_scaling",
        "coefficient": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
    }

    CALIBRATION_FILE.write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "project": "O-RAP",
        "target": "P(error > 1 km)",
        "error_threshold_km": 1.0,
        "features": FEATURES,
        "model_version": "orap-xgb-v1.0",
        "training_period": "2025-01 through 2025-08",
        "calibration_period": "2025-09 through 2025-10",
        "xgboost_configuration": {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 5,
            "random_state": 42,
        },
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {MODEL_FILE}")
    print(f"  {CALIBRATION_FILE}")
    print(f"  {METADATA_FILE}")
    print("\n✓ Production model artifacts created.")


if __name__ == "__main__":
    main()