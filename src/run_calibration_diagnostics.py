from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
RESULTS_DIR = BASE_DIR / "results" / "tables"
FIGURES_DIR = BASE_DIR / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "unreliable_1km"

SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]


# ============================================================
# FROZEN GLOBAL XGBOOST CONFIGURATION
# ============================================================

XGB_CONFIG = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
}


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "horizon_only": [
        "horizon_hours",
    ],

    "horizon_age": [
        "horizon_hours",
        "source_tle_age_hours",
    ],

    "full_orbital": [
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
    ],
}


# ============================================================
# HELPERS
# ============================================================

def train_xgb(train_df, features):
    X = train_df[features]
    y = train_df[TARGET]

    positives = int(y.sum())
    negatives = len(y) - positives

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=negatives / positives,
        **XGB_CONFIG,
    )

    model.fit(X, y)

    return model


def predict(model, df, features):
    return model.predict_proba(
        df[features]
    )[:, 1]


def fit_platt_calibrator(y, probabilities):
    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(logits, y)

    return calibrator


def apply_calibrator(calibrator, probabilities):
    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    logits = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


def calibration_bins(y, probabilities, n_bins=10):
    """
    Equal-width probability bins.

    For each bin:
        mean predicted probability
        observed positive frequency
        sample count
    """

    y = np.asarray(y)
    probabilities = np.asarray(probabilities)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": i,
                "bin_lower": lower,
                "bin_upper": upper,
                "mean_predicted": np.nan,
                "observed_frequency": np.nan,
                "count": 0,
            })
            continue

        rows.append({
            "bin": i,
            "bin_lower": lower,
            "bin_upper": upper,
            "mean_predicted": probabilities[mask].mean(),
            "observed_frequency": y[mask].mean(),
            "count": count,
        })

    return pd.DataFrame(rows)


def calculate_ece(y, probabilities, n_bins=10):
    """
    Expected Calibration Error.

    ECE = sum(
        bin_size / total_size *
        absolute(observed - predicted)
    )
    """

    table = calibration_bins(
        y,
        probabilities,
        n_bins,
    )

    total = len(y)

    ece = 0.0

    for _, row in table.iterrows():

        if row["count"] == 0:
            continue

        error = abs(
            row["observed_frequency"]
            - row["mean_predicted"]
        )

        ece += (
            row["count"] / total
        ) * error

    return ece


def calculate_mce(y, probabilities, n_bins=10):
    """
    Maximum Calibration Error.
    """

    table = calibration_bins(
        y,
        probabilities,
        n_bins,
    )

    errors = []

    for _, row in table.iterrows():

        if row["count"] == 0:
            continue

        errors.append(
            abs(
                row["observed_frequency"]
                - row["mean_predicted"]
            )
        )

    if not errors:
        return np.nan

    return max(errors)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("O-RAP — CALIBRATION DIAGNOSTICS")
print("=" * 70)

train = pd.read_csv(
    DATA_DIR / "train_temporal.csv"
)

validation = pd.read_csv(
    DATA_DIR / "validation_temporal.csv"
)

test = pd.read_csv(
    DATA_DIR / "test_temporal.csv"
)

print(f"Train:      {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


# ============================================================
# STORAGE
# ============================================================

all_test_predictions = []
summary_rows = []


# ============================================================
# RUN ALL THREE MODELS
# ============================================================

for model_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    for held_out in SATELLITES:

        print(
            f"\nHeld-out satellite: {held_out}"
        )

        fold_train = train[
            train["satellite_name"] != held_out
        ].copy()

        fold_validation = validation[
            validation["satellite_name"] != held_out
        ].copy()

        fold_test = test[
            test["satellite_name"] == held_out
        ].copy()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = train_xgb(
            fold_train,
            features,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_y = (
            fold_validation[TARGET]
            .to_numpy()
        )

        validation_raw = predict(
            model,
            fold_validation,
            features,
        )

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        calibrator = fit_platt_calibrator(
            validation_y,
            validation_raw,
        )

        validation_calibrated = apply_calibrator(
            calibrator,
            validation_raw,
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_y = (
            fold_test[TARGET]
            .to_numpy()
        )

        test_raw = predict(
            model,
            fold_test,
            features,
        )

        test_calibrated = apply_calibrator(
            calibrator,
            test_raw,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        raw_brier = brier_score_loss(
            test_y,
            test_raw,
        )

        calibrated_brier = brier_score_loss(
            test_y,
            test_calibrated,
        )

        raw_logloss = log_loss(
            test_y,
            test_raw,
            labels=[0, 1],
        )

        calibrated_logloss = log_loss(
            test_y,
            test_calibrated,
            labels=[0, 1],
        )

        raw_ece = calculate_ece(
            test_y,
            test_raw,
        )

        calibrated_ece = calculate_ece(
            test_y,
            test_calibrated,
        )

        raw_mce = calculate_mce(
            test_y,
            test_raw,
        )

        calibrated_mce = calculate_mce(
            test_y,
            test_calibrated,
        )

        print(
            f"  Raw ECE:        {raw_ece:.4f}"
        )

        print(
            f"  Calibrated ECE: {calibrated_ece:.4f}"
        )

        print(
            f"  Raw MCE:        {raw_mce:.4f}"
        )

        print(
            f"  Calibrated MCE: {calibrated_mce:.4f}"
        )

        print(
            f"  Brier: "
            f"{raw_brier:.4f} -> "
            f"{calibrated_brier:.4f}"
        )

        print(
            f"  LogLoss: "
            f"{raw_logloss:.4f} -> "
            f"{calibrated_logloss:.4f}"
        )

        # ----------------------------------------------------
        # SAVE TEST PREDICTIONS
        # ----------------------------------------------------

        prediction_df = pd.DataFrame({
            "model": model_name,
            "held_out_satellite": held_out,
            "y_true": test_y,
            "raw_probability": test_raw,
            "calibrated_probability": test_calibrated,
        })

        all_test_predictions.append(
            prediction_df
        )

        # ----------------------------------------------------
        # SAVE CALIBRATION TABLES
        # ----------------------------------------------------

        raw_bins = calibration_bins(
            test_y,
            test_raw,
        )

        raw_bins["model"] = model_name
        raw_bins["held_out_satellite"] = held_out
        raw_bins["calibration"] = "raw"

        calibrated_bins = calibration_bins(
            test_y,
            test_calibrated,
        )

        calibrated_bins["model"] = model_name
        calibrated_bins["held_out_satellite"] = held_out
        calibrated_bins["calibration"] = "calibrated"

        bins = pd.concat(
            [
                raw_bins,
                calibrated_bins,
            ],
            ignore_index=True,
        )

        bin_path = (
            RESULTS_DIR
            / f"calibration_bins_{model_name}_{held_out.replace(' ', '_').replace('-', '_')}.csv"
        )

        bins.to_csv(
            bin_path,
            index=False,
        )

        summary_rows.append({
            "model": model_name,
            "held_out_satellite": held_out,
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "raw_logloss": raw_logloss,
            "calibrated_logloss": calibrated_logloss,
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "raw_mce": raw_mce,
            "calibrated_mce": calibrated_mce,
        })


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_test_predictions,
    ignore_index=True,
)

predictions_path = (
    RESULTS_DIR
    / "calibration_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    RESULTS_DIR
    / "calibration_diagnostics_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# POOLED UNSEEN-SATELLITE CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("POOLED UNSEEN-SATELLITE CALIBRATION")
print("=" * 70)

pooled_rows = []

for model_name in FEATURE_SETS:

    subset = predictions_df[
        predictions_df["model"] == model_name
    ]

    y = subset["y_true"].to_numpy()

    raw = subset[
        "raw_probability"
    ].to_numpy()

    calibrated = subset[
        "calibrated_probability"
    ].to_numpy()

    raw_ece = calculate_ece(
        y,
        raw,
    )

    calibrated_ece = calculate_ece(
        y,
        calibrated,
    )

    raw_mce = calculate_mce(
        y,
        raw,
    )

    calibrated_mce = calculate_mce(
        y,
        calibrated,
    )

    raw_brier = brier_score_loss(
        y,
        raw,
    )

    calibrated_brier = brier_score_loss(
        y,
        calibrated,
    )

    raw_logloss = log_loss(
        y,
        raw,
        labels=[0, 1],
    )

    calibrated_logloss = log_loss(
        y,
        calibrated,
        labels=[0, 1],
    )

    pooled_rows.append({
        "model": model_name,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_logloss": raw_logloss,
        "calibrated_logloss": calibrated_logloss,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "raw_mce": raw_mce,
        "calibrated_mce": calibrated_mce,
    })

    print(f"\n{model_name}")

    print(
        f"  Brier: "
        f"{raw_brier:.4f} -> "
        f"{calibrated_brier:.4f}"
    )

    print(
        f"  LogLoss: "
        f"{raw_logloss:.4f} -> "
        f"{calibrated_logloss:.4f}"
    )

    print(
        f"  ECE: "
        f"{raw_ece:.4f} -> "
        f"{calibrated_ece:.4f}"
    )

    print(
        f"  MCE: "
        f"{raw_mce:.4f} -> "
        f"{calibrated_mce:.4f}"
    )


pooled_df = pd.DataFrame(
    pooled_rows
)

pooled_df.to_csv(
    RESULTS_DIR
    / "pooled_calibration_results.csv",
    index=False,
)


# ============================================================
# RELIABILITY DIAGRAM — POOLED
# ============================================================

print("\n" + "=" * 70)
print("GENERATING POOLED RELIABILITY DIAGRAMS")
print("=" * 70)


def plot_reliability(
    y,
    probabilities,
    title,
    output_path,
):

    table = calibration_bins(
        y,
        probabilities,
        n_bins=10,
    )

    valid = table[
        table["count"] > 0
    ]

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        valid["mean_predicted"],
        valid["observed_frequency"],
        marker="o",
        label="Model",
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed frequency"
    )

    plt.title(title)

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


for model_name in FEATURE_SETS:

    subset = predictions_df[
        predictions_df["model"] == model_name
    ]

    y = subset["y_true"].to_numpy()

    raw = subset[
        "raw_probability"
    ].to_numpy()

    calibrated = subset[
        "calibrated_probability"
    ].to_numpy()

    plot_reliability(
        y,
        raw,
        f"O-RAP Reliability Diagram — {model_name} — Raw",
        FIGURES_DIR
        / f"reliability_{model_name}_raw.png",
    )

    plot_reliability(
        y,
        calibrated,
        f"O-RAP Reliability Diagram — {model_name} — Calibrated",
        FIGURES_DIR
        / f"reliability_{model_name}_calibrated.png",
    )


# ============================================================
# FULL ORBITAL — PER-SATELLITE DIAGRAMS
# ============================================================

print(
    "\nGenerating per-satellite full-orbital diagrams..."
)

for satellite in SATELLITES:

    subset = predictions_df[
        (predictions_df["model"] == "full_orbital")
        &
        (
            predictions_df[
                "held_out_satellite"
            ] == satellite
        )
    ]

    y = subset["y_true"].to_numpy()

    raw = subset[
        "raw_probability"
    ].to_numpy()

    calibrated = subset[
        "calibrated_probability"
    ].to_numpy()

    safe_name = (
        satellite
        .replace(" ", "_")
        .replace("-", "_")
    )

    plot_reliability(
        y,
        raw,
        f"Full O-RAP Reliability — {satellite} — Raw",
        FIGURES_DIR
        / f"reliability_full_orbital_{safe_name}_raw.png",
    )

    plot_reliability(
        y,
        calibrated,
        f"Full O-RAP Reliability — {satellite} — Calibrated",
        FIGURES_DIR
        / f"reliability_full_orbital_{safe_name}_calibrated.png",
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION DIAGNOSTICS COMPLETE")
print("=" * 70)

print("\nSaved tables:")
print(
    f"  {summary_path}"
)

print(
    f"  {RESULTS_DIR / 'pooled_calibration_results.csv'}"
)

print(
    f"  {predictions_path}"
)

print("\nSaved figures:")
print(
    f"  {FIGURES_DIR}"
)

print("\n✓ Calibration diagnostics completed.")