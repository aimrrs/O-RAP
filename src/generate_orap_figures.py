
"""
O-RAP Publication Figure Generator
===================================

Generates six publication-quality figures from existing experiment outputs.

Reproducibility:
- No experimental results are hardcoded.
- All plotted observations are loaded from experiment CSV files.
- Derived metrics are calculated from the loaded data.
- Missing files or required columns cause an explicit error.
- SHA256 hashes and source metadata are recorded in a manifest.
- No model training or test-set fitting is performed.

Output:
results/figures/
    figure_01_chronological_curves.png/.pdf
    figure_02_calibration.png/.pdf
    figure_03_feature_importance.png/.pdf
    figure_04_loso_pr_auc.png/.pdf
    figure_05_loso_roc_auc.png/.pdf
    figure_06_threshold_sensitivity.png/.pdf
    figure_generation_manifest.json
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    ROOT / "results" / "modeling" / "experiments"
)

ORBITAL_DIR = EXPERIMENT_DIR / "orbital_only_analysis"
LOSO_DIR = EXPERIMENT_DIR / "loso"
THRESHOLD_DIR = EXPERIMENT_DIR / "threshold_sensitivity"

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 12,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# REPRODUCIBILITY TRACKING
# ============================================================

MANIFEST = {
    "project": "O-RAP",
    "generator": "src/generate_orap_figures.py",
    "generated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "matplotlib_version": plt.matplotlib.__version__,
    "figure_resolution_dpi": DPI,
    "figures": [],
    "input_files": [],
}


def sha256_file(path):
    """Calculate SHA256 hash of an input file."""

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def register_input(path, columns):
    """Record input file and its hash in the manifest."""

    path = Path(path)

    MANIFEST["input_files"].append({
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "columns_used": columns,
    })


def register_figure(name, source_files, description):
    """Record generated figure and its source files."""

    MANIFEST["figures"].append({
        "filename": name,
        "source_files": source_files,
        "description": description,
        "formats": ["PNG", "PDF"],
        "resolution_dpi": DPI,
    })


# ============================================================
# GENERAL HELPERS
# ============================================================

def read_csv(path):
    """Load CSV and fail if the file is unavailable."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required experiment file not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(
            f"Experiment file is empty:\n{path}"
        )

    return rows


def resolve_column(rows, candidates, description):
    """
    Resolve a required column from explicitly supported aliases.
    Never invent or substitute a value.
    """

    available = set(rows[0].keys())

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Required column missing: {description}\n"
        f"Accepted names: {candidates}\n"
        f"Available columns: {sorted(available)}"
    )


def to_float(value, description):
    """Convert a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid numeric value for {description}: {value}"
        )

    if not np.isfinite(result):
        raise ValueError(
            f"Non-finite value for {description}: {value}"
        )

    return result


def to_int(value, description):
    """Convert a value to an integer."""

    number = to_float(value, description)

    if not number.is_integer():
        raise ValueError(
            f"Expected integer for {description}: {value}"
        )

    return int(number)


def save_figure(fig, filename, source_files, description):
    """Save PNG and vector PDF versions."""

    png_path = OUTPUT_DIR / f"{filename}.png"
    pdf_path = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    register_figure(
        filename,
        source_files,
        description,
    )

    print(f"[SAVED] {png_path.name}")
    print(f"[SAVED] {pdf_path.name}")


def validate_binary_labels(y, description):
    """Ensure both target classes are represented."""

    classes = np.unique(y)

    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(
            f"{description} must contain both binary classes 0 and 1. "
            f"Found: {classes}"
        )


# ============================================================
# FIGURE 1: CHRONOLOGICAL PR AND ROC CURVES
# ============================================================

def plot_chronological_curves():

    path = ORBITAL_DIR / "orbital_only_test_predictions.csv"
    rows = read_csv(path)

    y_col = resolve_column(
        rows,
        ["actual_label", "label_1km", "y_true", "actual"],
        "chronological actual labels",
    )

    p_col = resolve_column(
        rows,
        [
            "calibrated_probability",
            "calibrated_prob",
            "platt_probability",
            "probability_calibrated",
        ],
        "chronological calibrated probabilities",
    )

    y = np.asarray([
        to_int(row[y_col], y_col)
        for row in rows
    ])

    p = np.asarray([
        to_float(row[p_col], p_col)
        for row in rows
    ])

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Chronological labels must be binary.")

    validate_binary_labels(y, "Chronological labels")

    ap = average_precision_score(y, p)
    roc_auc = roc_auc_score(y, p)
    prevalence = float(np.mean(y))

    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
    )

    # Precision-recall panel.
    ax = axes[0]

    ax.plot(
        recall,
        precision,
        linewidth=1.8,
        label=f"Orbital-only XGBoost (AP={ap:.3f})",
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.1,
        label=f"Positive prevalence={prevalence:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(a) Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    # ROC panel.
    ax = axes[1]

    ax.plot(
        fpr,
        tpr,
        linewidth=1.8,
        label=f"Orbital-only XGBoost (AUC={roc_auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.1,
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(b) ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.22)

    fig.suptitle(
        "O-RAP Chronological Test Performance",
        fontweight="bold",
    )

    fig.tight_layout()

    register_input(
        path,
        [y_col, p_col],
    )

    save_figure(
        fig,
        "figure_01_chronological_curves",
        [str(path.relative_to(ROOT))],
        "Chronological precision-recall and ROC curves.",
    )

    print(
        f"[INFO] Chronological AP={ap:.6f}, "
        f"ROC-AUC={roc_auc:.6f}, "
        f"prevalence={prevalence:.6f}"
    )


# ============================================================
# FIGURE 2: CALIBRATION DIAGRAM
# ============================================================

def plot_calibration():

    path = ORBITAL_DIR / "orbital_only_calibration_bins.csv"
    rows = read_csv(path)

    predicted_col = resolve_column(
        rows,
        [
            "mean_predicted_probability",
            "predicted_probability",
            "mean_predicted",
            "predicted",
        ],
        "calibration mean predicted probability",
    )

    observed_col = resolve_column(
        rows,
        [
            "observed_positive_rate",
            "observed_rate",
            "fraction_of_positives",
            "observed",
        ],
        "calibration observed positive rate",
    )

    count_col = resolve_column(
        rows,
        [
            "count",
            "samples",
            "n",
            "sample_count",
        ],
        "calibration bin sample count",
    )

    predicted = np.asarray([
        to_float(row[predicted_col], predicted_col)
        for row in rows
    ])

    observed = np.asarray([
        to_float(row[observed_col], observed_col)
        for row in rows
    ])

    counts = np.asarray([
        to_int(row[count_col], count_col)
        for row in rows
    ])

    if np.any(counts <= 0):
        raise ValueError("Calibration bin counts must be positive.")

    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("Predicted probabilities must be in [0,1].")

    if np.any((observed < 0) | (observed > 1)):
        raise ValueError("Observed rates must be in [0,1].")

    order = np.argsort(predicted)

    predicted = predicted[order]
    observed = observed[order]
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(5.4, 4.5))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.3,
        label="Perfect calibration",
    )

    sizes = 30 + 100 * counts / np.max(counts)

    ax.scatter(
        predicted,
        observed,
        s=sizes,
        zorder=3,
        label="Orbital-only XGBoost",
    )

    ax.plot(
        predicted,
        observed,
        linewidth=1,
        alpha=0.65,
    )

    for x, y, n in zip(predicted, observed, counts):
        ax.annotate(
            f"N={n:,}",
            (x, y),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6.5,
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Frequency")
    ax.set_title("O-RAP Probability Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.22)

    fig.tight_layout()

    register_input(
        path,
        [predicted_col, observed_col, count_col],
    )

    save_figure(
        fig,
        "figure_02_calibration",
        [str(path.relative_to(ROOT))],
        "Calibration diagram with actual sample counts per bin.",
    )


# ============================================================
# FIGURE 3: ORBITAL FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance():

    path = ORBITAL_DIR / "orbital_only_feature_importance.csv"
    rows = read_csv(path)

    feature_col = resolve_column(
        rows,
        ["feature", "feature_name"],
        "feature name",
    )

    importance_col = resolve_column(
        rows,
        ["importance", "feature_importance", "gain"],
        "feature importance",
    )

    names = [row[feature_col] for row in rows]

    values = np.asarray([
        to_float(row[importance_col], importance_col)
        for row in rows
    ])

    if np.any(values < 0):
        raise ValueError("Feature importance cannot be negative.")

    order = np.argsort(values)

    names = [names[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    bars = ax.barh(
        names,
        values,
        height=0.68,
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Orbital Feature")
    ax.set_title("Orbital-Only XGBoost Feature Importance")
    ax.grid(axis="x", alpha=0.22)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            xy=(
                value,
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )

    fig.tight_layout()

    register_input(
        path,
        [feature_col, importance_col],
    )

    save_figure(
        fig,
        "figure_03_feature_importance",
        [str(path.relative_to(ROOT))],
        "Source orbital feature importance from the trained model.",
    )


# ============================================================
# FIGURES 4 AND 5: LOSO SATELLITE PERFORMANCE
# ============================================================

def plot_loso_satellites():

    path = LOSO_DIR / "loso_by_satellite.csv"
    rows = read_csv(path)

    name_col = resolve_column(
        rows,
        ["satellite_name", "held_out_satellite"],
        "held-out satellite name",
    )

    pr_col = resolve_column(
        rows,
        ["raw_pr_auc"],
        "raw LOSO average precision",
    )

    roc_col = resolve_column(
        rows,
        ["raw_roc_auc"],
        "raw LOSO ROC-AUC",
    )

    n_col = resolve_column(
        rows,
        ["test_samples"],
        "held-out satellite test sample count",
    )

    pos_col = resolve_column(
        rows,
        ["test_positives"],
        "held-out satellite positive count",
    )

    names = [row[name_col] for row in rows]

    samples = np.asarray([
        to_int(row[n_col], n_col)
        for row in rows
    ])

    positives = np.asarray([
        to_int(row[pos_col], pos_col)
        for row in rows
    ])

    pr_values = np.asarray([
        to_float(row[pr_col], pr_col)
        for row in rows
    ])

    roc_values = np.asarray([
        to_float(row[roc_col], roc_col)
        for row in rows
    ])

    if np.any(samples <= 0):
        raise ValueError("LOSO test sample counts must be positive.")

    if np.any(positives < 0) or np.any(positives > samples):
        raise ValueError("Invalid LOSO positive counts.")

    if np.any((pr_values < 0) | (pr_values > 1)):
        raise ValueError("LOSO PR-AUC values must be in [0,1].")

    if np.any((roc_values < 0) | (roc_values > 1)):
        raise ValueError("LOSO ROC-AUC values must be in [0,1].")

    order = np.argsort(pr_values)

    names = [names[i] for i in order]
    samples = samples[order]
    positives = positives[order]
    pr_values = pr_values[order]
    roc_values = roc_values[order]

    source = str(path.relative_to(ROOT))

    # PR-AUC figure.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    bars = ax.barh(
        names,
        pr_values,
        height=0.68,
    )

    ax.set_xlabel("Average Precision (PR-AUC)")
    ax.set_ylabel("Held-Out Satellite")
    ax.set_title("LOSO Precision-Recall Performance")
    ax.set_xlim(
        0,
        min(1.0, max(pr_values) + 0.35),
    )
    ax.grid(axis="x", alpha=0.22)

    for bar, n, pos in zip(bars, samples, positives):
        ax.annotate(
            f"N={n:,}, Positive={pos:,}",
            xy=(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )

    fig.tight_layout()

    register_input(
        path,
        [name_col, pr_col, n_col, pos_col],
    )

    save_figure(
        fig,
        "figure_04_loso_pr_auc",
        [source],
        "Raw per-fold LOSO average precision with test sample and positive counts.",
    )

    # ROC-AUC figure.
    roc_order = np.argsort(roc_values)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    bars = ax.barh(
        [names[i] for i in roc_order],
        roc_values[roc_order],
        height=0.68,
    )

    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Held-Out Satellite")
    ax.set_title("LOSO ROC Performance")
    ax.set_xlim(
        0,
        min(1.0, max(roc_values) + 0.35),
    )
    ax.grid(axis="x", alpha=0.22)

    for bar, i in zip(bars, roc_order):
        ax.annotate(
            f"N={samples[i]:,}, Positive={positives[i]:,}",
            xy=(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_05_loso_roc_auc",
        [source],
        "Raw per-fold LOSO ROC-AUC with test sample and positive counts.",
    )

    print(
        "[INFO] LOSO plots use raw per-fold ranking metrics. "
        "Fold-specific calibration is not pooled."
    )


# ============================================================
# FIGURE 6: THRESHOLD SENSITIVITY
# ============================================================

def plot_threshold_sensitivity():

    path = (
        THRESHOLD_DIR
        / "threshold_sensitivity_summary.csv"
    )

    rows = read_csv(path)

    threshold_col = resolve_column(
        rows,
        ["disagreement_threshold_km"],
        "disagreement threshold",
    )

    pr_col = resolve_column(
        rows,
        ["calibrated_pr_auc"],
        "calibrated PR-AUC",
    )

    roc_col = resolve_column(
        rows,
        ["calibrated_roc_auc"],
        "calibrated ROC-AUC",
    )

    f1_col = resolve_column(
        rows,
        ["calibrated_f1"],
        "calibrated F1-score",
    )

    samples_col = resolve_column(
        rows,
        ["calibrated_samples"],
        "calibrated test sample count",
    )

    positives_col = resolve_column(
        rows,
        ["calibrated_positives"],
        "calibrated test positive count",
    )

    thresholds = np.asarray([
        to_float(row[threshold_col], threshold_col)
        for row in rows
    ])

    pr = np.asarray([
        to_float(row[pr_col], pr_col)
        for row in rows
    ])

    roc = np.asarray([
        to_float(row[roc_col], roc_col)
        for row in rows
    ])

    f1 = np.asarray([
        to_float(row[f1_col], f1_col)
        for row in rows
    ])

    samples = np.asarray([
        to_int(row[samples_col], samples_col)
        for row in rows
    ])

    positives = np.asarray([
        to_int(row[positives_col], positives_col)
        for row in rows
    ])

    if np.any(samples <= 0):
        raise ValueError("Threshold test sample counts must be positive.")

    if np.any(positives < 0) or np.any(positives > samples):
        raise ValueError("Invalid threshold positive counts.")

    if np.any((pr < 0) | (pr > 1)):
        raise ValueError("Threshold PR-AUC must be in [0,1].")

    if np.any((roc < 0) | (roc > 1)):
        raise ValueError("Threshold ROC-AUC must be in [0,1].")

    if np.any((f1 < 0) | (f1 > 1)):
        raise ValueError("Threshold F1 must be in [0,1].")

    order = np.argsort(thresholds)

    thresholds = thresholds[order]
    pr = pr[order]
    roc = roc[order]
    f1 = f1[order]
    samples = samples[order]
    positives = positives[order]

    # Calculate prevalence from actual test counts.
    prevalence = 100.0 * positives / samples

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 5.8),
    )

    plots = [
        (
            axes[0, 0],
            pr,
            "PR-AUC",
            "(a) Precision-Recall",
        ),
        (
            axes[0, 1],
            roc,
            "ROC-AUC",
            "(b) ROC",
        ),
        (
            axes[1, 0],
            f1,
            "F1-score",
            "(c) F1-score",
        ),
        (
            axes[1, 1],
            prevalence,
            "Positive rate (%)",
            "(d) Positive-class prevalence",
        ),
    ]

    for ax, values, ylabel, title in plots:

        ax.plot(
            thresholds,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
        )

        for x, y in zip(thresholds, values):

            label = (
                f"{y:.2f}%"
                if ylabel == "Positive rate (%)"
                else f"{y:.3f}"
            )

            ax.annotate(
                label,
                (x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

        ax.set_xlabel("Disagreement Threshold (km)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(thresholds)
        ax.grid(alpha=0.22)

    fig.suptitle(
        "O-RAP Sensitivity to Position-Disagreement Threshold",
        fontweight="bold",
    )

    fig.tight_layout()

    register_input(
        path,
        [
            threshold_col,
            pr_col,
            roc_col,
            f1_col,
            samples_col,
            positives_col,
        ],
    )

    save_figure(
        fig,
        "figure_06_threshold_sensitivity",
        [str(path.relative_to(ROOT))],
        "Threshold sensitivity using measured test metrics and prevalence derived from counts.",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("O-RAP PUBLICATION FIGURE GENERATION")
    print("Strict data-driven reproducible version")
    print("=" * 72)

    print("\n[1/6] Chronological PR and ROC curves")
    plot_chronological_curves()

    print("\n[2/6] Calibration diagram")
    plot_calibration()

    print("\n[3/6] Feature importance")
    plot_feature_importance()

    print("\n[4/6] LOSO PR-AUC")
    print_loso_status = plot_loso_satellites()

    print("\n[5/6] LOSO ROC-AUC")
    print_loso_status = print_loso_status

    print("\n[6/6] Threshold sensitivity")
    plot_threshold_sensitivity()

    manifest_path = (
        OUTPUT_DIR / "figure_generation_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            MANIFEST,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("FIGURE GENERATION COMPLETED")
    print("=" * 72)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    print(
        f"Figures generated: {len(MANIFEST['figures'])}"
    )

    print(
        f"Input files tracked: {len(MANIFEST['input_files'])}"
    )

    print("[PASS] No experimental values hardcoded.")
    print("[PASS] All figure data sourced from experiment outputs.")
    print("[PASS] Input file SHA256 hashes recorded.")
    print("[PASS] PNG and PDF versions generated.")
    print("[PASS] No model retraining performed.")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:
        print("\n[ERROR] Figure generation stopped.")
        print(str(exc))
        raise
