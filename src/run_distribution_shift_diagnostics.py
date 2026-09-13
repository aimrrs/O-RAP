from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# O-RAP — ISS DISTRIBUTION SHIFT DIAGNOSTICS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data" / "processed" / "splits" / "train_temporal.csv"
VAL_FILE = BASE_DIR / "data" / "processed" / "splits" / "validation_temporal.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "splits" / "test_temporal.csv"

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HELD_OUT_SATELLITE = "ISS"

FEATURES = [
    "semimajor_axis",
    "mean_motion",
    "inclination",
    "eccentricity",
    "bstar",
    "source_tle_age_hours",
    "horizon_hours",
    "apoapsis",
    "periapsis",
]

DISPLAY_NAMES = {
    "semimajor_axis": "Semimajor axis",
    "mean_motion": "Mean motion",
    "inclination": "Inclination",
    "eccentricity": "Eccentricity",
    "bstar": "BSTAR",
    "source_tle_age_hours": "TLE age",
    "horizon_hours": "Prediction horizon",
    "apoapsis": "Apoapsis",
    "periapsis": "Periapsis",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def empirical_ks_distance(x, y):
    """
    Two-sample Kolmogorov-Smirnov distance calculated without scipy.
    """
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))

    combined = np.sort(np.unique(np.concatenate([x, y])))

    cdf_x = np.searchsorted(x, combined, side="right") / len(x)
    cdf_y = np.searchsorted(y, combined, side="right") / len(y)

    return float(np.max(np.abs(cdf_x - cdf_y)))


def standardized_mean_difference(x, y):
    """
    Standardized mean difference:
        (mean_y - mean_x) / pooled_std

    Absolute value is used for magnitude.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_sd = np.sqrt((var_x + var_y) / 2.0)

    if pooled_sd == 0:
        return 0.0

    return float((mean_y - mean_x) / pooled_sd)


def summarize_feature(reference, target):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    target = pd.to_numeric(target, errors="coerce").dropna()

    smd = standardized_mean_difference(reference, target)
    ks = empirical_ks_distance(reference, target)

    return {
        "reference_n": len(reference),
        "target_n": len(target),
        "reference_mean": reference.mean(),
        "target_mean": target.mean(),
        "reference_std": reference.std(),
        "target_std": target.std(),
        "reference_median": reference.median(),
        "target_median": target.median(),
        "smd": smd,
        "abs_smd": abs(smd),
        "ks_distance": ks,
    }


def print_shift_interpretation(abs_smd):
    if abs_smd < 0.10:
        return "negligible"
    elif abs_smd < 0.25:
        return "small"
    elif abs_smd < 0.50:
        return "moderate"
    else:
        return "large"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("=" * 70)
print("O-RAP — ISS DISTRIBUTION SHIFT DIAGNOSTICS")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train:      {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test:       {len(test)}")


# ------------------------------------------------------------
# Define populations
#
# Reference population:
#   all NON-ISS satellites in validation
#
# Target population:
#   ISS in temporal test
#
# Additional reference:
#   all NON-ISS satellites in temporal train
# ------------------------------------------------------------

calibration_population = validation[
    validation["satellite_name"] != HELD_OUT_SATELLITE
].copy()

iss_test = test[
    test["satellite_name"] == HELD_OUT_SATELLITE
].copy()

training_population = train[
    train["satellite_name"] != HELD_OUT_SATELLITE
].copy()

print("\nPopulation definitions:")
print(f"  Non-ISS training:      {len(training_population)}")
print(f"  Non-ISS validation:    {len(calibration_population)}")
print(f"  ISS test:              {len(iss_test)}")


if len(iss_test) == 0:
    raise RuntimeError("No ISS samples found in the test split.")


# ------------------------------------------------------------
# Failure prevalence
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET / FAILURE PREVALENCE SHIFT")
print("=" * 70)

for name, df in [
    ("Non-ISS training", training_population),
    ("Non-ISS validation", calibration_population),
    ("ISS test", iss_test),
]:
    rate = df["unreliable_1km"].mean()
    positives = int(df["unreliable_1km"].sum())

    print(
        f"{name:22s}: "
        f"{positives:4d}/{len(df):4d} "
        f"= {rate:.4%}"
    )


# ------------------------------------------------------------
# Feature distribution comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NON-ISS VALIDATION → ISS TEST DISTRIBUTION SHIFT")
print("=" * 70)

rows = []

for feature in FEATURES:
    result = summarize_feature(
        calibration_population[feature],
        iss_test[feature],
    )

    result["feature"] = feature
    result["display_name"] = DISPLAY_NAMES[feature]
    result["interpretation"] = print_shift_interpretation(
        result["abs_smd"]
    )

    rows.append(result)

shift_df = pd.DataFrame(rows)

shift_df = shift_df[
    [
        "feature",
        "display_name",
        "reference_n",
        "target_n",
        "reference_mean",
        "target_mean",
        "reference_std",
        "target_std",
        "reference_median",
        "target_median",
        "smd",
        "abs_smd",
        "ks_distance",
        "interpretation",
    ]
]

shift_df = shift_df.sort_values(
    "abs_smd",
    ascending=False,
)

print()

for _, row in shift_df.iterrows():
    print(
        f"{row['display_name']:22s} "
        f"SMD={row['smd']:+.3f} "
        f"| KS={row['ks_distance']:.3f} "
        f"| {row['interpretation']}"
    )

shift_file = TABLE_DIR / "iss_distribution_shift.csv"
shift_df.to_csv(shift_file, index=False)


# ------------------------------------------------------------
# Training → ISS comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NON-ISS TRAINING → ISS TEST")
print("=" * 70)

train_rows = []

for feature in FEATURES:
    result = summarize_feature(
        training_population[feature],
        iss_test[feature],
    )

    result["feature"] = feature
    result["display_name"] = DISPLAY_NAMES[feature]
    result["interpretation"] = print_shift_interpretation(
        result["abs_smd"]
    )

    train_rows.append(result)

train_shift_df = pd.DataFrame(train_rows)

train_shift_df = train_shift_df.sort_values(
    "abs_smd",
    ascending=False,
)

for _, row in train_shift_df.iterrows():
    print(
        f"{row['display_name']:22s} "
        f"SMD={row['smd']:+.3f} "
        f"| KS={row['ks_distance']:.3f} "
        f"| {row['interpretation']}"
    )

train_shift_file = TABLE_DIR / "iss_train_to_test_shift.csv"
train_shift_df.to_csv(train_shift_file, index=False)


# ------------------------------------------------------------
# Horizon distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION HORIZON DISTRIBUTION")
print("=" * 70)

ref_horizon = (
    calibration_population["horizon_hours"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

iss_horizon = (
    iss_test["horizon_hours"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

horizon_df = pd.DataFrame({
    "non_iss_validation_percent": ref_horizon,
    "iss_test_percent": iss_horizon,
}).fillna(0)

print(horizon_df.round(2))

horizon_file = TABLE_DIR / "iss_horizon_distribution.csv"
horizon_df.to_csv(horizon_file)


# ------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING DISTRIBUTION PLOTS")
print("=" * 70)

# 3 x 3 grid: one figure containing all nine feature comparisons.
fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 12),
)

axes = axes.flatten()

for ax, feature in zip(axes, FEATURES):

    ref = pd.to_numeric(
        calibration_population[feature],
        errors="coerce",
    ).dropna()

    target = pd.to_numeric(
        iss_test[feature],
        errors="coerce",
    ).dropna()

    # Use percentile clipping only for visualization.
    # Statistics above are computed on the complete data.
    combined = pd.concat([ref, target])

    lower = combined.quantile(0.01)
    upper = combined.quantile(0.99)

    ref_plot = ref[(ref >= lower) & (ref <= upper)]
    target_plot = target[(target >= lower) & (target <= upper)]

    ax.hist(
        ref_plot,
        bins=30,
        alpha=0.55,
        density=True,
        label="Non-ISS validation",
    )

    ax.hist(
        target_plot,
        bins=30,
        alpha=0.55,
        density=True,
        label="ISS test",
    )

    row = shift_df[
        shift_df["feature"] == feature
    ].iloc[0]

    ax.set_title(
        f"{DISPLAY_NAMES[feature]}\n"
        f"SMD={row['smd']:+.2f}, KS={row['ks_distance']:.2f}"
    )

    ax.set_xlabel(DISPLAY_NAMES[feature])
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.suptitle(
    "O-RAP: Non-ISS Calibration Population vs Unseen ISS Test",
    fontsize=16,
)

plt.tight_layout()

distribution_plot = (
    FIGURE_DIR / "iss_distribution_shift.png"
)

plt.savefig(
    distribution_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Boxplot comparison
# ------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    3,
    figsize=(14, 12),
)

axes = axes.flatten()

for ax, feature in zip(axes, FEATURES):

    ref = pd.to_numeric(
        calibration_population[feature],
        errors="coerce",
    ).dropna()

    target = pd.to_numeric(
        iss_test[feature],
        errors="coerce",
    ).dropna()

    combined = pd.concat([ref, target])

    lower = combined.quantile(0.01)
    upper = combined.quantile(0.99)

    ref_plot = ref[(ref >= lower) & (ref <= upper)]
    target_plot = target[(target >= lower) & (target <= upper)]

    ax.boxplot(
        [ref_plot, target_plot],
        tick_labels=["Non-ISS", "ISS"],
    )

    row = shift_df[
        shift_df["feature"] == feature
    ].iloc[0]

    ax.set_title(
        f"{DISPLAY_NAMES[feature]}\n"
        f"|SMD|={row['abs_smd']:.2f}"
    )

    ax.set_ylabel(DISPLAY_NAMES[feature])

plt.suptitle(
    "O-RAP: Feature Distribution Shift",
    fontsize=16,
)

plt.tight_layout()

boxplot_file = (
    FIGURE_DIR / "iss_distribution_shift_boxplots.png"
)

plt.savefig(
    boxplot_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Shift ranking figure
# ------------------------------------------------------------

ranked = shift_df.sort_values(
    "abs_smd",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    ranked["display_name"],
    ranked["abs_smd"],
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

ax.set_xlabel("Absolute standardized mean difference")
ax.set_title(
    "ISS Distribution Shift Relative to Non-ISS Calibration Population"
)

ax.legend()

plt.tight_layout()

shift_rank_file = (
    FIGURE_DIR / "iss_distribution_shift_ranking.png"
)

plt.savefig(
    shift_rank_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LARGEST DISTRIBUTION SHIFTS")
print("=" * 70)

top = shift_df.head(5)

for _, row in top.iterrows():
    print(
        f"  {row['display_name']:22s} "
        f"| SMD={row['smd']:+.3f} "
        f"| KS={row['ks_distance']:.3f}"
    )


print("\nSaved tables:")
print(f"  {shift_file}")
print(f"  {train_shift_file}")
print(f"  {horizon_file}")

print("\nSaved figures:")
print(f"  {distribution_plot}")
print(f"  {boxplot_file}")
print(f"  {shift_rank_file}")

print("\n" + "=" * 70)
print("DISTRIBUTION SHIFT DIAGNOSTICS COMPLETE")
print("=" * 70)