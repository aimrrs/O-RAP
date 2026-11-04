
from pathlib import Path
import csv
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_tables"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "model_comparison": ROOT / "results/modeling/experiments/model_comparison.csv",
    "orbital_metrics": ROOT / "results/modeling/experiments/orbital_only_analysis/orbital_only_calibration_metrics.csv",
    "loso": ROOT / "results/modeling/experiments/loso/loso_by_satellite.csv",
    "loso_pooled": ROOT / "results/modeling/experiments/loso/loso_pooled_metrics.json",
    "threshold": ROOT / "results/modeling/experiments/threshold_sensitivity/threshold_sensitivity_summary.csv",
    "splits": ROOT / "results/modeling/splits/split_summary.csv",
    "boundary_excluded": ROOT / "results/modeling/splits/boundary_excluded_pairs.csv",
}

REQUIRED_COLUMNS = {
    "model_comparison": [
        "model", "partition", "samples", "positives",
        "positive_rate_percent", "pr_auc", "roc_auc",
        "precision", "recall", "f1", "brier_score",
        "log_loss", "decision_threshold"
    ],
    "orbital_metrics": [
        "model", "samples", "positives", "positive_rate_percent",
        "pr_auc", "roc_auc", "brier_score", "log_loss",
        "precision", "recall", "f1", "threshold"
    ],
    "loso": [
        "satellite_name", "test_samples", "test_positives",
        "raw_pr_auc", "raw_roc_auc",
        "calibrated_brier_score", "calibrated_log_loss",
        "calibrated_f1"
    ],
    "threshold": [
        "disagreement_threshold_km", "calibrated_samples",
        "calibrated_positives", "calibrated_positive_rate_percent",
        "calibrated_pr_auc", "calibrated_roc_auc",
        "calibrated_brier_score", "calibrated_log_loss",
        "calibrated_precision", "calibrated_recall",
        "calibrated_f1", "calibrated_threshold"
    ],
    "splits": [
        "partition", "satellite_name", "pair_count",
        "positive_count", "negative_count"
    ],
    "boundary_excluded": [
        "pair_id", "label_1km"
    ],
}


def read_csv(name):
    path = SOURCES[name]

    if not path.exists():
        raise FileNotFoundError(f"Required source missing: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    missing = set(REQUIRED_COLUMNS[name]) - set(columns)
    if missing:
        raise ValueError(
            f"{name}: missing required columns: {sorted(missing)}"
        )

    return rows


def read_json(name):
    path = SOURCES[name]

    if not path.exists():
        raise FileNotFoundError(f"Required source missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def number(value):
    return float(value)


def integer(value):
    return int(float(value))


def fmt(value, digits=4):
    return f"{number(value):.{digits}f}"


def write_table(name, headers, rows, note=None):
    csv_path = OUT / f"{name}.csv"
    md_path = OUT / f"{name}.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")

        if note:
            f.write("\n" + note + "\n")

    print(f"[SAVED] {csv_path.name}")
    print(f"[SAVED] {md_path.name}")


def main():
    print("=" * 72)
    print("O-RAP FINAL IEEE RESULTS TABLE GENERATION")
    print("=" * 72)

    source_rows = {}

    for name in REQUIRED_COLUMNS:
        source_rows[name] = read_csv(name)
        print(f"[LOADED] {name}: {len(source_rows[name]):,} rows")

    pooled = read_json("loso_pooled")
    print("[LOADED] loso_pooled_metrics.json")

    # TABLE 1: Dataset split distribution, including boundary exclusions
    split_rows = source_rows["splits"]
    split_order = ["train", "validation", "test"]
    table = []

    for partition in split_order:
        subset = [
            r for r in split_rows
            if r["partition"].lower() == partition
        ]

        samples = sum(integer(r["pair_count"]) for r in subset)
        positives = sum(integer(r["positive_count"]) for r in subset)
        negatives = sum(integer(r["negative_count"]) for r in subset)
        rate = 100 * positives / samples if samples else 0

        table.append([
            partition.capitalize(),
            samples,
            positives,
            negatives,
            f"{rate:.3f}%"
        ])

    excluded_count = len(source_rows["boundary_excluded"])

    table.append([
        "Boundary-excluded pairs",
        excluded_count,
        "—",
        "—",
        "—",
    ])

    write_table(
        "table_01_dataset_split",
        ["Partition", "Samples", "Positive", "Negative", "Positive rate"],
        table,
        note=(
            f"Note: {excluded_count} pairs were excluded because their "
            "source and reference TLE epochs belonged to different "
            "chronological partitions."
        )
    )

    # TABLE 2: Model comparison, test partition only
    model_rows = [
        r for r in source_rows["model_comparison"]
        if r["partition"].lower() == "test"
    ]

    table = []

    for r in model_rows:
        table.append([
            r["model"],
            integer(r["samples"]),
            fmt(r["pr_auc"]),
            fmt(r["roc_auc"]),
            fmt(r["f1"]),
            fmt(r["brier_score"]),
            fmt(r["log_loss"]),
        ])

    write_table(
        "table_02_model_comparison",
        ["Model", "N", "PR-AUC", "ROC-AUC", "F1",
         "Brier score", "Log loss"],
        table
    )

    # TABLE 3: Orbital-only raw vs Platt calibration
    table = []

    for r in source_rows["orbital_metrics"]:
        table.append([
            r["model"],
            integer(r["samples"]),
            integer(r["positives"]),
            f'{number(r["positive_rate_percent"]):.3f}%',
            fmt(r["pr_auc"]),
            fmt(r["roc_auc"]),
            fmt(r["brier_score"]),
            fmt(r["log_loss"]),
            fmt(r["precision"]),
            fmt(r["recall"]),
            fmt(r["f1"]),
            fmt(r["threshold"]),
        ])

    write_table(
        "table_03_orbital_calibration",
        ["Model", "N", "Positive", "Prevalence", "PR-AUC",
         "ROC-AUC", "Brier", "Log loss", "Precision",
         "Recall", "F1", "Decision threshold"],
        table
    )

    # TABLE 4: LOSO per-satellite results
    table = []

    for r in source_rows["loso"]:
        table.append([
            r["satellite_name"],
            integer(r["test_samples"]),
            integer(r["test_positives"]),
            fmt(r["raw_pr_auc"]),
            fmt(r["raw_roc_auc"]),
            fmt(r["calibrated_brier_score"]),
            fmt(r["calibrated_log_loss"]),
            fmt(r["calibrated_f1"]),
        ])

    write_table(
        "table_04_loso_satellite",
        ["Held-out satellite", "N", "Positive", "Raw PR-AUC",
         "Raw ROC-AUC", "Calibrated Brier", "Calibrated log loss",
         "Calibrated F1"],
        table
    )

    # TABLE 5: Pooled LOSO metrics
    # Fold-specific calibrators can change score ordering.
    # Therefore, do not report pooled calibrated ranking metrics.
    pooled_rows = [
        ["Samples", pooled["samples"]],
        ["Positive samples", pooled["positives"]],
        ["Positive rate (%)", fmt(pooled["positive_rate_percent"], 3)],
        ["Raw PR-AUC", fmt(pooled["raw_pr_auc"])],
        ["Raw ROC-AUC", fmt(pooled["raw_roc_auc"])],
        ["Calibrated Brier score", fmt(pooled["calibrated_brier_score"])],
        ["Calibrated log loss", fmt(pooled["calibrated_log_loss"])],
        ["Calibrated precision", fmt(pooled["calibrated_precision"])],
        ["Calibrated recall", fmt(pooled["calibrated_recall"])],
        ["Calibrated F1", fmt(pooled["calibrated_f1"])],
    ]

    write_table(
        "table_05_loso_pooled",
        ["Metric", "Value"],
        pooled_rows,
        note=(
            "Note: Each held-out satellite fold used its own Platt "
            "calibrator. Pooled calibrated Brier score and log loss "
            "summarize per-sample probability errors, but calibrated "
            "PR-AUC and ROC-AUC are omitted because fold-specific "
            "calibration can change score ordering across folds. "
            "Raw pooled ranking metrics are reported instead."
        )
    )

    # TABLE 6: Disagreement threshold sensitivity
    threshold_rows = sorted(
        source_rows["threshold"],
        key=lambda r: number(r["disagreement_threshold_km"])
    )

    table = []

    for r in threshold_rows:
        prevalence = number(r["calibrated_positive_rate_percent"]) / 100

        table.append([
            f'> {number(r["disagreement_threshold_km"]):g} km',
            integer(r["calibrated_samples"]),
            integer(r["calibrated_positives"]),
            f"{prevalence * 100:.3f}%",
            f"{prevalence:.4f}",
            fmt(r["calibrated_pr_auc"]),
            fmt(r["calibrated_roc_auc"]),
            fmt(r["calibrated_brier_score"]),
            fmt(r["calibrated_log_loss"]),
            fmt(r["calibrated_precision"]),
            fmt(r["calibrated_recall"]),
            fmt(r["calibrated_f1"]),
        ])

    write_table(
        "table_06_threshold_sensitivity",
        [
            "Label threshold", "N", "Positive", "Prevalence",
            "PR-AUC baseline", "PR-AUC", "ROC-AUC", "Brier",
            "Log loss", "Precision", "Recall", "F1"
        ],
        table,
        note=(
            "Note: The PR-AUC baseline is the positive-class prevalence "
            "for each threshold. PR-AUC should be interpreted relative "
            "to this threshold-specific baseline."
        )
    )

    # REPRODUCIBILITY MANIFEST: hashes of all source files
    manifest = {
        "description": "O-RAP final IEEE results tables",
        "source_files": {},
        "generated_tables": [],
    }

    for name, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest source missing: {path}"
            )

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest["source_files"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
        }

    for path in sorted(OUT.glob("table_*")):
        if path.name != "table_generation_manifest.json":
            manifest["generated_tables"].append(path.name)

    manifest_path = OUT / "table_generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("TABLE GENERATION COMPLETED")
    print("=" * 72)
    print(f"Output directory: {OUT}")
    print("Tables generated: 6 (CSV + Markdown)")
    print("[PASS] Boundary-excluded pairs reported separately.")
    print("[PASS] Pooled LOSO calibrated ranking metrics omitted.")
    print("[PASS] Threshold-specific prevalence baselines included.")
    print("[PASS] Source SHA256 hashes recorded.")
    print("[PASS] No model retraining performed.")
    print("[PASS] No experimental metric values hardcoded.")


if __name__ == "__main__":
    main()
