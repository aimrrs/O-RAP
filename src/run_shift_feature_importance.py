from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


# ============================================================
# O-RAP — SHIFT-AWARE FEATURE IMPORTANCE DIAGNOSTICS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "train_temporal.csv"
)

VAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "validation_temporal.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "test_temporal.csv"
)

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

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

DISPLAY_NAMES = {
    "horizon_hours": "Prediction horizon",
    "source_tle_age_hours": "TLE age",
    "mean_motion": "Mean motion",
    "eccentricity": "Eccentricity",
    "inclination": "Inclination",
    "ra_of_asc_node": "RA of ascending node",
    "arg_of_pericenter": "Argument of pericenter",
    "mean_anomaly": "Mean anomaly",
    "bstar": "BSTAR",
    "mean_motion_dot": "Mean motion dot",
    "mean_motion_ddot": "Mean motion d-dot",
    "semimajor_axis": "Semimajor axis",
    "period": "Period",
    "apoapsis": "Apoapsis",
    "periapsis": "Periapsis",
}

SATELLITES = [
    "GPS BIIR-2",
    "Hubble Space Telescope",
    "ISS",
    "NOAA 19",
    "Sentinel-1A",
    "TDRS-5",
]

# Frozen global configuration selected earlier.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# Helpers
# ============================================================

def train_model(train_df):
    """
    Train the frozen global XGBoost configuration.

    scale_pos_weight is calculated from the actual training
    population for each leave-one-satellite-out fold.
    """

    X = train_df[FEATURES]
    y = train_df["unreliable_1km"].astype(int)

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    if positives == 0:
        raise RuntimeError("Training set contains no positive samples.")

    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X, y)

    return model, scale_pos_weight


def get_importance_table(model, held_out_satellite):
    """
    Extract gain-based XGBoost feature importance.

    Gain is used because it represents the average improvement
    in the objective contributed by splits using each feature.
    """

    importance = model.get_booster().get_score(
        importance_type="gain"
    )

    rows = []

    for feature in FEATURES:
        gain = importance.get(feature, 0.0)

        rows.append({
            "held_out_satellite": held_out_satellite,
            "feature": feature,
            "display_name": DISPLAY_NAMES[feature],
            "gain": float(gain),
        })

    result = pd.DataFrame(rows)

    total_gain = result["gain"].sum()

    if total_gain > 0:
        result["normalized_gain"] = (
            result["gain"] / total_gain
        )
    else:
        result["normalized_gain"] = 0.0

    result = result.sort_values(
        "normalized_gain",
        ascending=False,
    )

    return result


def calculate_distribution_shift(
    reference_df,
    target_df,
):
    """
    Calculate SMD and KS distance without scipy.
    """

    rows = []

    for feature in FEATURES:

        reference = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        target = pd.to_numeric(
            target_df[feature],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        mean_ref = np.mean(reference)
        mean_target = np.mean(target)

        std_ref = np.std(reference, ddof=1)
        std_target = np.std(target, ddof=1)

        pooled_sd = np.sqrt(
            (std_ref ** 2 + std_target ** 2) / 2.0
        )

        if pooled_sd == 0:
            smd = 0.0
        else:
            smd = (
                mean_target - mean_ref
            ) / pooled_sd

        reference_sorted = np.sort(reference)
        target_sorted = np.sort(target)

        combined = np.sort(
            np.unique(
                np.concatenate(
                    [reference_sorted, target_sorted]
                )
            )
        )

        cdf_ref = (
            np.searchsorted(
                reference_sorted,
                combined,
                side="right",
            )
            / len(reference_sorted)
        )

        cdf_target = (
            np.searchsorted(
                target_sorted,
                combined,
                side="right",
            )
            / len(target_sorted)
        )

        ks = np.max(
            np.abs(cdf_ref - cdf_target)
        )

        rows.append({
            "feature": feature,
            "display_name": DISPLAY_NAMES[feature],
            "smd": float(smd),
            "abs_smd": float(abs(smd)),
            "ks_distance": float(ks),
            "reference_mean": float(mean_ref),
            "target_mean": float(mean_target),
            "reference_std": float(std_ref),
            "target_std": float(std_target),
        })

    return pd.DataFrame(rows).sort_values(
        "abs_smd",
        ascending=False,
    )


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("O-RAP — SHIFT-AWARE FEATURE IMPORTANCE DIAGNOSTICS")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train:      {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


# ============================================================
# Leave-one-satellite-out analysis
# ============================================================

all_importance = []
all_shift = []

print("\n" + "=" * 70)
print("LEAVE-ONE-SATELLITE-OUT FEATURE ANALYSIS")
print("=" * 70)

for held_out in SATELLITES:

    print("\n" + "-" * 70)
    print(f"HELD-OUT SATELLITE: {held_out}")
    print("-" * 70)

    train_nonheld = train[
        train["satellite_name"] != held_out
    ].copy()

    val_nonheld = validation[
        validation["satellite_name"] != held_out
    ].copy()

    test_held = test[
        test["satellite_name"] == held_out
    ].copy()

    print(
        f"Training population:   {len(train_nonheld)}"
    )
    print(
        f"Calibration population:{len(val_nonheld)}"
    )
    print(
        f"Held-out test:         {len(test_held)}"
    )

    # --------------------------------------------------------
    # Train frozen model
    # --------------------------------------------------------

    model, scale_pos_weight = train_model(
        train_nonheld
    )

    print(
        f"Scale-pos-weight:      "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = get_importance_table(
        model,
        held_out,
    )

    all_importance.append(
        importance_df
    )

    print("\nTop model features:")

    for _, row in importance_df.head(8).iterrows():

        print(
            f"  {row['display_name']:24s} "
            f"{row['normalized_gain']:.4f}"
        )

    # --------------------------------------------------------
    # Distribution shift:
    #
    # Calibration population → held-out test
    # --------------------------------------------------------

    shift_df = calculate_distribution_shift(
        val_nonheld,
        test_held,
    )

    shift_df["held_out_satellite"] = held_out

    all_shift.append(
        shift_df
    )

    print("\nLargest feature shifts:")

    for _, row in shift_df.head(8).iterrows():

        print(
            f"  {row['display_name']:24s} "
            f"SMD={row['smd']:+.3f} "
            f"| KS={row['ks_distance']:.3f}"
        )


# ============================================================
# Combine results
# ============================================================

importance_all = pd.concat(
    all_importance,
    ignore_index=True,
)

shift_all = pd.concat(
    all_shift,
    ignore_index=True,
)


# ============================================================
# Save raw tables
# ============================================================

importance_file = (
    TABLE_DIR
    / "leave_one_satellite_feature_importance.csv"
)

shift_file = (
    TABLE_DIR
    / "leave_one_satellite_distribution_shift.csv"
)

importance_all.to_csv(
    importance_file,
    index=False,
)

shift_all.to_csv(
    shift_file,
    index=False,
)


# ============================================================
# Mean importance across held-out satellites
# ============================================================

mean_importance = (
    importance_all
    .groupby(
        ["feature", "display_name"],
        as_index=False,
    )["normalized_gain"]
    .mean()
    .sort_values(
        "normalized_gain",
        ascending=False,
    )
)

mean_importance_file = (
    TABLE_DIR
    / "mean_leave_one_satellite_feature_importance.csv"
)

mean_importance.to_csv(
    mean_importance_file,
    index=False,
)


# ============================================================
# ISS-specific comparison:
#
# Distribution shift magnitude
#       vs
# Model importance
# ============================================================

iss_importance = importance_all[
    importance_all["held_out_satellite"] == "ISS"
].copy()

iss_shift = shift_all[
    shift_all["held_out_satellite"] == "ISS"
].copy()

iss_combined = pd.merge(
    iss_shift[
        [
            "feature",
            "display_name",
            "smd",
            "abs_smd",
            "ks_distance",
        ]
    ],
    iss_importance[
        [
            "feature",
            "normalized_gain",
        ]
    ],
    on="feature",
    how="inner",
)

iss_combined = iss_combined.sort_values(
    "normalized_gain",
    ascending=False,
)

iss_file = (
    TABLE_DIR
    / "iss_shift_vs_feature_importance.csv"
)

iss_combined.to_csv(
    iss_file,
    index=False,
)


# ============================================================
# Rank correlation between shift and importance
# ============================================================

# Spearman correlation implemented using pandas ranks.
rank_shift = iss_combined["abs_smd"].rank()
rank_importance = iss_combined["normalized_gain"].rank()

spearman = rank_shift.corr(
    rank_importance,
    method="pearson",
)

print("\n" + "=" * 70)
print("ISS SHIFT vs MODEL IMPORTANCE")
print("=" * 70)

print(
    f"Spearman rank correlation "
    f"(absolute SMD vs feature importance): "
    f"{spearman:.4f}"
)

print("\nISS feature comparison:")

for _, row in iss_combined.sort_values(
    "abs_smd",
    ascending=False,
).iterrows():

    print(
        f"  {row['display_name']:24s} "
        f"| shift SMD={row['smd']:+.3f} "
        f"| KS={row['ks_distance']:.3f} "
        f"| model gain={row['normalized_gain']:.4f}"
    )


# ============================================================
# Plot 1:
# ISS distribution shift vs model importance
# ============================================================

plot_df = iss_combined.copy()

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.scatter(
    plot_df["abs_smd"],
    plot_df["normalized_gain"],
)

for _, row in plot_df.iterrows():

    ax.annotate(
        row["display_name"],
        (
            row["abs_smd"],
            row["normalized_gain"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel(
    "Absolute standardized mean difference"
)

ax.set_ylabel(
    "Normalized XGBoost gain"
)

ax.set_title(
    "ISS: Feature Distribution Shift vs Model Importance"
)

plt.tight_layout()

iss_scatter_file = (
    FIGURE_DIR
    / "iss_shift_vs_feature_importance.png"
)

plt.savefig(
    iss_scatter_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Plot 2:
# Top model features across all held-out satellites
# ============================================================

pivot_importance = importance_all.pivot(
    index="display_name",
    columns="held_out_satellite",
    values="normalized_gain",
)

# Order rows by overall mean importance.
ordered_features = (
    mean_importance["display_name"]
    .tolist()
)

pivot_importance = pivot_importance.reindex(
    ordered_features
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

im = ax.imshow(
    pivot_importance.fillna(0).to_numpy(),
    aspect="auto",
)

ax.set_yticks(
    np.arange(len(pivot_importance.index))
)

ax.set_yticklabels(
    pivot_importance.index
)

ax.set_xticks(
    np.arange(len(pivot_importance.columns))
)

ax.set_xticklabels(
    pivot_importance.columns,
    rotation=45,
    ha="right",
)

ax.set_title(
    "O-RAP Feature Importance Across Held-Out Satellites"
)

fig.colorbar(
    im,
    ax=ax,
    label="Normalized XGBoost gain",
)

plt.tight_layout()

importance_heatmap_file = (
    FIGURE_DIR
    / "leave_one_satellite_feature_importance_heatmap.png"
)

plt.savefig(
    importance_heatmap_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Plot 3:
# ISS distribution shift ranking
# ============================================================

iss_rank = iss_shift.sort_values(
    "abs_smd",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    iss_rank["display_name"],
    iss_rank["abs_smd"],
)

ax.axvline(
    0.10,
    linestyle="--",
    label="0.10",
)

ax.axvline(
    0.25,
    linestyle="--",
    label="0.25",
)

ax.axvline(
    0.50,
    linestyle="--",
    label="0.50",
)

ax.set_xlabel(
    "Absolute standardized mean difference"
)

ax.set_title(
    "ISS Feature Distribution Shift"
)

ax.legend()

plt.tight_layout()

iss_shift_file = (
    FIGURE_DIR
    / "iss_feature_shift_ranking.png"
)

plt.savefig(
    iss_shift_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Summary table:
# features with both high shift and high importance
# ============================================================

shift_threshold = 0.50
importance_threshold = 0.05

iss_combined["high_shift"] = (
    iss_combined["abs_smd"]
    >= shift_threshold
)

iss_combined["high_importance"] = (
    iss_combined["normalized_gain"]
    >= importance_threshold
)

iss_combined["both_high"] = (
    iss_combined["high_shift"]
    & iss_combined["high_importance"]
)

both_high = iss_combined[
    iss_combined["both_high"]
].copy()

both_high = both_high.sort_values(
    "normalized_gain",
    ascending=False,
)

both_high_file = (
    TABLE_DIR
    / "iss_high_shift_high_importance_features.csv"
)

both_high.to_csv(
    both_high_file,
    index=False,
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("FEATURES WITH BOTH LARGE ISS SHIFT AND HIGH MODEL IMPORTANCE")
print("=" * 70)

if len(both_high) == 0:

    print(
        "No feature simultaneously exceeded "
        f"|SMD| >= {shift_threshold:.2f} and "
        f"importance >= {importance_threshold:.2f}."
    )

else:

    for _, row in both_high.iterrows():

        print(
            f"  {row['display_name']:24s} "
            f"| SMD={row['smd']:+.3f} "
            f"| KS={row['ks_distance']:.3f} "
            f"| gain={row['normalized_gain']:.4f}"
        )


print("\n" + "=" * 70)
print("SAVED TABLES")
print("=" * 70)

print(f"  {importance_file}")
print(f"  {shift_file}")
print(f"  {mean_importance_file}")
print(f"  {iss_file}")
print(f"  {both_high_file}")


print("\n" + "=" * 70)
print("SAVED FIGURES")
print("=" * 70)

print(f"  {iss_scatter_file}")
print(f"  {importance_heatmap_file}")
print(f"  {iss_shift_file}")


print("\n" + "=" * 70)
print("SHIFT-AWARE FEATURE IMPORTANCE COMPLETE")
print("=" * 70)