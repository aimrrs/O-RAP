# Paper-aligned results guide

This folder contains aggregate results and metadata for the current O-RAP manuscript. Raw TLEs and derived row-level test predictions are not included here.

## Folders

- `nested_loso/`: corrected nested satellite-and-time holdout results, per-satellite metrics, inner selections, calibration parameters, paired ranking differences, flag-all F1, Brier-skill diagnostics, and reliability figure.
- `chronological/`: purged chronological test metrics, comparators, thresholds, and run metadata.

## Reproduction

From the repository root, use the commands in `README.md`. The nested runner creates local prediction files to compute summaries and the reliability figure; Git ignores those row-level files. The committed aggregate files are reproducibility snapshots and do not contain raw TLE rows.

## Metric interpretation

- ROC-AUC and average precision are macro-averaged across the five folds with both classes.
- F1 and probability-error means are macro-averaged across all six satellite folds; F1 uses `zero_division=0`.
- The flag-all baseline is a fixed rule, not a model fitted to test labels.
- Chronological pooled metrics and within-satellite macro metrics are both reported because satellite prevalence differs.

## Scope

The target is agreement between two SGP4-propagated TLE states at a common target epoch, not independent physical truth. Results cover six satellites from 2025 and do not establish operational calibration or population-level generalization.
