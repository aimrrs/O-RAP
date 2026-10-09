# O-RAP: Orbital Reliability Assessment and Prediction

O-RAP evaluates whether information available from a Two-Line Element (TLE) and a requested SGP4 propagation can rank cases where the propagated position differs from a later-TLE reference by more than a selected tolerance. The reference is a **TLE-agreement proxy**, not independently measured position truth. This repository contains source code, processed inputs, historical results, and a separate paper-aligned reproduction.

## Current paper-aligned findings

The corrected manuscript uses raw XGBoost as its primary risk-ranking score and treats Platt calibration as a secondary diagnostic. In nested satellite-and-time holdout, the six-fold macro results are:

| Method | ROC-AUC* | Average precision* | Macro F1** |
|---|---:|---:|---:|
| XGBoost, raw | 0.8538 | 0.3895 | 0.1964 |
| Logistic regression, raw | 0.8662 | 0.4950 | 0.1774 |
| Age + horizon | 0.8681 | 0.4763 | 0.3074 |
| Horizon only | 0.8304 | 0.3543 | 0.3120 |
| Flag-all rule | — | — | 0.2270 |

*Ranking metrics average over the five two-class folds. **F1 averages all six folds, with zero for folds without a positive prediction or positive case.** The full fold-level aggregate is in `results/experiments/paper_final/nested_loso/`.

The separate purged chronological test contains 2,595 samples and 418 positives. Raw XGBoost obtains pooled ROC-AUC/AP/F1 of 0.9240/0.6379/0.5893; raw logistic regression obtains 0.9110/0.6757/0.6468. Within-satellite macro AUC/AP are included in the chronological results CSV to show the effect of pooling satellites with different event rates.

These are exploratory results from six satellites and one year. They do not establish operational utility or generalization to other objects, years, or arbitrary stale-TLE query conditions.

## Data provenance and sharing

The checked-in `data/raw/` records are Space-Track GP/GP_History exports; processed labels and features derive from those records. These raw files are already tracked in the repository and will remain in Git history on the pushed branch. Check the current terms applicable to the account and intended public distribution before publishing or mirroring them: [Space-Track documentation](https://www.space-track.org/documentation) and [Space-Track User Agreement](https://www.space-track.org/auth/createAccount). If redistribution is not permitted, do not treat deleting files in a later commit as erasing their earlier Git history. The code and source data have different rights. No code license is declared in this repository; do not infer permission to reuse code from the presence of the data or vice versa.

A suggested data citation is: *Space-Track.org, General Perturbations and GP_History orbital data, accessed [insert actual access date], https://www.space-track.org.* Include any acknowledgment required by the applicable terms.

## Label construction and coordinate handling

`src/build_multisatellite_labels.py` selects the first later TLE created at or after the forecast target, subject to the maximum reference-delay limit. Both source and reference TLEs are propagated by SGP4 to the same target time. The Euclidean distance between their TEME position vectors is stored as `proxy_error_km`; no frame transformation is applied because both vectors use the same frame and epoch.

The requested prediction horizons are 6, 12, 24, and 48 hours. The baseline maximum reference delay is 6 hours **after the target time**; it is separate from the forecast horizon. The default label is positive when `proxy_error_km > 1.0`. The sensitivity scripts vary the error tolerance (0.5, 1, 2, 5 km) and reference-delay cap (1, 2, 4, 6 h). Reference timestamps and proxy-error fields are excluded from model features.

Six labels are excluded because two distinct ISS source TLEs share the same source-event key (NORAD 25544; creation time 2025-01-25 19:06:24 UTC; epoch 2025-01-25 04:55:56.509824 UTC). There are two ambiguous labels at each of 6, 24, and 48 h. The event is excluded rather than matched arbitrarily. The feature set contains 17,263 samples from ISS, Hubble, NOAA 19, Sentinel-1A, TDRS-5, and GPS BIIR-2.

## Evaluation protocols

### Nested satellite-and-time holdout

Each outer fold holds out one satellite's November–December data. XGBoost configuration and logistic-regression regularization are selected using the other satellites. Cross-satellite out-of-fold development margins fit Platt scaling and select F1 thresholds. The final models train on January–August rows from the five development satellites. Rows whose reference creation time crosses the September or November boundary are purged. Outer test labels are used only for final scoring. The run records 23 purged training rows, 9 purged validation rows, and 2,595 test rows.

### Chronological holdout

The chronological run trains on January–August, validates on September–October, and tests on November–December. It applies the same reference-availability purging. Logistic-regression `C` is selected by five forward-chaining folds inside training; probability and heuristic thresholds are selected on validation data. Test scores are raw and uncalibrated.

### Legacy sensitivity analyses

The original sensitivity experiments are preserved under `results/experiments/sensitivity/`. They use the earlier pooled-validation protocol and change the retained population as tolerance or delay changes. Treat them as descriptive legacy analyses, not as primary nested estimates in the current paper.

## Repository layout

```text
data/raw/                              Space-Track source exports
 data/processed/                       Labels, features, temporal splits
src/                                   Data builders and existing experiment scripts
src/reproducibility/                   Paper-aligned corrected experiment scripts
results/tables/                        Previously saved project tables (preserved)
results/figures/                       Previously saved project figures (preserved)
results/experiments/                   Historical experiments and paper_final results
```

`venv/` and `.venv/` are local environments and are ignored by Git. `requirements.txt` pins the installed project dependencies.

## Environment setup

The verified project environment is Python 3.14.7 with NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1, SGP4 2.27, and Matplotlib 3.11.2. On Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproduce the paper-aligned experiments

Run from the repository root using the processed inputs in `data/processed/`:

```powershell
$py = ".\.venv\Scripts\python.exe"
$results = "results\experiments\paper_final"

& $py src\reproducibility\run_nested_loso_final.py data\processed "$results\nested_loso"
& $py src\reproducibility\summarize_final_nested.py "$results\nested_loso"
& $py src\reproducibility\report_decision_diagnostics.py "$results\nested_loso" "$results\nested_loso"
& $py src\reproducibility\run_temporal_purged_validation.py data\processed "$results\chronological"
```

The nested runner writes the aggregate tables and metadata plus row-level prediction files needed to regenerate diagnostics. The prediction files are ignored by Git and are not included in the committed paper results. The summary and decision scripts generate the calibration plot, calibration parameters, flag-all baseline, Brier-skill tables, and paired fold differences.

The paper-aligned metadata records the Python/package versions, random seed, selected settings, purged-row counts, and SHA-256 hashes of the processed model inputs. See `results/experiments/paper_final/README.md` for the result-file guide and reproduction caveats.

## Reproduce legacy sensitivity results

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe src\run_sensitivity_experiments.py
.\venv\Scripts\python.exe src\aggregate_sensitivity_results.py
```

The legacy sensitivity outputs are written under `results/experiments/sensitivity/` and should be interpreted with their sample sizes, class balance, and older protocol in view.

## Live Space-Track access

Download, coverage, and live verification scripts require Space-Track credentials. Set `SPACE_TRACK_IDENTITY` and `SPACE_TRACK_PASSWORD` in the process environment. Never put credentials in source files or commit `.env` files. A credential was present in an older Git commit; rotate it if that has not already been done. The offline experiments do not need account credentials.

## Citation and reuse

Cite the associated paper and the Space-Track source data when reusing results. This repository currently has no explicit code license. Authors should choose and add an appropriate license before inviting code reuse. The paper-aligned outputs are research artifacts, not an operational orbit or conjunction product.
