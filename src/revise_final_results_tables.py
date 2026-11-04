
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_tables"
SPLIT_FILE = ROOT / "results/modeling/splits/boundary_excluded_pairs.csv"
POOLED_FILE = ROOT / "results/modeling/experiments/loso/loso_pooled_metrics.json"
THRESHOLD_FILE = ROOT / "results/modeling/experiments/threshold_sensitivity/threshold_sensitivity_summary.csv"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_table(name, headers, rows, note=None):
    with (OUT / f"{name}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    with (OUT / f"{name}.md").open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
        if note:
            f.write("\n" + note + "\n")

    print(f"[UPDATED] {name}")


def main():
    print("O-RAP FINAL TABLE CORRECTIONS")

    # Table 1: record boundary exclusions separately.
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    excluded_count = sum(1 for _ in read_csv(SPLIT_FILE))

    rows = read_csv(OUT / "table_01_dataset_split.csv")
    table = [
        [r["Partition"], r["Samples"], r["Positive"],
         r["Negative"], r["Positive rate"]]
        for r in rows
    ]
    table.append([
        "Boundary-excluded pairs",
        excluded_count,
        "—", "—", "—"
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

    # Table 5: do not imply pooled fold-calibrated ranking scores
    # are globally comparable across satellite folds.
    with POOLED_FILE.open("r", encoding="utf-8") as f:
        p = json.load(f)

    table = [
        ["Samples", p["samples"]],
        ["Positive samples", p["positives"]],
        ["Positive rate (%)", f'{p["positive_rate_percent"]:.3f}'],
        ["Raw PR-AUC", f'{p["raw_pr_auc"]:.4f}'],
        ["Raw ROC-AUC", f'{p["raw_roc_auc"]:.4f}'],
        ["Calibrated Brier score", f'{p["calibrated_brier_score"]:.4f}'],
        ["Calibrated log loss", f'{p["calibrated_log_loss"]:.4f}'],
        ["Calibrated precision", f'{p["calibrated_precision"]:.4f}'],
        ["Calibrated recall", f'{p["calibrated_recall"]:.4f}'],
        ["Calibrated F1", f'{p["calibrated_f1"]:.4f}'],
    ]

    write_table(
        "table_05_loso_pooled",
        ["Metric", "Value"],
        table,
        note=(
            "Note: Each held-out satellite fold used its own Platt "
            "calibrator. Pooled calibrated Brier score and log loss "
            "summarize per-sample probability errors, but calibrated "
            "PR-AUC and ROC-AUC are omitted because fold-specific "
            "calibration can change score ordering across folds. "
            "Raw pooled ranking metrics are reported instead."
        )
    )

    # Table 6: add positive prevalence as the PR-AUC reference baseline.
    threshold_rows = read_csv(THRESHOLD_FILE)
    threshold_rows.sort(
        key=lambda r: float(r["disagreement_threshold_km"])
    )

    table = []
    for r in threshold_rows:
        prevalence = float(r["calibrated_positive_rate_percent"]) / 100
        pr_auc = float(r["calibrated_pr_auc"])

        table.append([
            f'> {float(r["disagreement_threshold_km"]):g} km',
            int(r["calibrated_samples"]),
            int(r["calibrated_positives"]),
            f"{prevalence * 100:.3f}%",
            f"{prevalence:.4f}",
            f"{pr_auc:.4f}",
            f"{float(r['calibrated_roc_auc']):.4f}",
            f"{float(r['calibrated_brier_score']):.4f}",
            f"{float(r['calibrated_log_loss']):.4f}",
            f"{float(r['calibrated_precision']):.4f}",
            f"{float(r['calibrated_recall']):.4f}",
            f"{float(r['calibrated_f1']):.4f}",
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

    print("\n[PASS] Table 1 includes boundary exclusions.")
    print("[PASS] Table 5 avoids pooled calibrated ranking comparisons.")
    print("[PASS] Table 6 includes prevalence baselines.")
    print("[PASS] No model retraining performed.")
    print(f"Updated tables: {OUT}")


if __name__ == "__main__":
    main()
