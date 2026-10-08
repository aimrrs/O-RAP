# O-RAP — Orbital Reliability Assessment and Prediction

O-RAP studies whether features available from a source Two-Line Element set (TLE) can estimate whether an SGP4-propagated position will diverge from a later TLE-derived proxy by more than a chosen distance. It is a research evaluation, not an operational orbit product.

## Research scope and limitations

The checked-in dataset covers six objects and source records from 2025: ISS (NORAD 25544), Hubble Space Telescope (20580), NOAA 19 (33591), Sentinel-1A (39634), TDRS-5 (24876), and GPS BIIR-2 (26407). The target is a **TLE-to-TLE proxy difference**, not independently measured position error or physical ground truth. Results should not be interpreted as conjunction assessment or safety guarantees.

Prediction horizons are **6, 12, 24, and 48 hours**. The reference-delay limit is a separate quantity: the baseline label builder accepts the first later TLE created at or after the target time only when that TLE arrives within **6 hours after the target**. The stored labels have a maximum `reference_delay_hours` of 6. The sensitivity study varies that maximum delay at 1, 2, 4, and 6 hours.

The default binary target is `unreliable_1km = 1` when `proxy_error_km > 1.0`; the threshold experiment repeats this definition at 0.5, 2, and 5 km. Reference creation/epoch, reference delay, target time, and proxy error are excluded from model features because they are not known at prediction time.

## Data provenance and sharing

The raw per-object CSV files in `data/raw/` are Space-Track GP/GP_History exports; processed labels and features are derived from those records. Cite Space-Track.org when reusing these data or analyses. Space-Track's documentation says redistribution of basic SSA information, including TLEs, is approved subject to appropriate citation, while its user agreement also describes prior express approval for transfers. Review the current terms applicable to your account and intended release before redistributing the raw exports. See [Space-Track documentation: SSA Sharing and Orbital Data Requests](https://www.space-track.org/documentation) and [Space-Track User Agreement](https://www.space-track.org/auth/createAccount).

A suggested data citation is: *Space-Track.org, General Perturbations and GP_History orbital data, accessed 2025, https://www.space-track.org.* Add the actual download/access date and any required acknowledgment to publications.

## Label construction and coordinate handling

`src/build_multisatellite_labels.py` selects the earliest record whose `CREATION_DATE` is at or after the forecast target, subject to the maximum reference delay. Both source and later-reference TLEs are propagated with SGP4 to the same target time; the Euclidean norm of their position difference is stored in `proxy_error_km`. These are proxy labels only.

The raw exports report `REF_FRAME=TEME` and `TIME_SYSTEM=UTC`. SGP4 returns position vectors in the TLE/SGP4 frame (TEME) and kilometres. The current label builder subtracts the two vectors directly because both are propagated to the same instant in the same frame. It performs no frame conversion. If future inputs use different frames or time systems, convert them to a common frame and epoch before comparing positions.

Feature matching found one ambiguous source event: ISS, `CREATION_DATE=2025-01-25 19:06:24 UTC`, `EPOCH=2025-01-25 04:55:56.509824 UTC`. Two distinct TLE line pairs share this key. The pipeline conservatively excludes the event rather than choosing one arbitrarily. Its 6-, 24-, and 48-hour labels (two rows per horizon, six rows total) are omitted, leaving 17,263 feature rows from 17,269 generated labels. The excluded TLE records remain in the raw data; the exclusion is applied when constructing features and splits.

## Evaluation protocol

The temporal split uses source creation month: January–August for training, September–October for validation, and November–December for test. The checked-in splits contain 12,657, 2,011, and 2,595 rows, respectively.

The LOSO implementation evaluates each satellite's November–December rows as that fold's test set. Training and final calibration/threshold fitting for the fold use the other five satellites. A global XGBoost configuration is selected by pooling validation predictions from all six folds. Consequently, a given satellite's September–October rows can contribute to global configuration selection through the other folds, even though its November–December test labels are never used for configuration, calibration, or threshold selection. This distinction is retained in the reproduction to match the original experiment and should be considered when interpreting the satellite-generalization estimate.

For each fold, Platt calibration is fitted to the logits of the raw probabilities using that fold's non-held-out validation rows. The F1 decision threshold is selected on the calibrated validation probabilities. The exported `calibration_coefficient`, `calibration_intercept`, and `threshold` columns in the per-satellite LOSO table therefore refer to that fold; they are not a single pooled coefficient. Test metrics are computed only after those choices are fixed.

## Repository layout

```text
data/raw/                         Space-Track source exports
 data/processed/                  Labels, features, and temporal splits
src/                              Data builders and experiment scripts
results/tables/                   Previously saved project results (preserved)
results/figures/                  Previously saved project figures (preserved)
results/experiments/              Reproductions, sensitivity inputs, outputs, and metadata
```

The `venv/` directory is local and ignored. `requirements.txt` records the packages installed in that project environment.

## Environment setup

The experiments were reproduced using Python 3.14.7 and the checked-in `requirements.txt` versions (including XGBoost 3.4.1, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, SGP4 2.27, and Matplotlib 3.11.2).

On Windows PowerShell, create a fresh virtual environment and install the recorded dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or use the existing project environment with `.\venv\Scripts\python.exe`.

## Reproduction commands

Run commands from the repository root. Rebuild the derived datasets in this order when needed:

```powershell
.\venv\Scripts\python.exe src\build_multisatellite_labels.py
.\venv\Scripts\python.exe src\build_features.py
.\venv\Scripts\python.exe src\make_splits.py
.\venv\Scripts\python.exe src\audit_data_exclusions.py
```

Reproduce the temporal model and final tuned LOSO experiment without overwriting the older tables:

```powershell
$env:ORAP_RESULTS_DIR = "$PWD\results\experiments\temporal_reproduction\tables"
.\venv\Scripts\python.exe src\run_xgboost.py
$env:ORAP_RESULTS_DIR = "$PWD\results\experiments\loso_reproduction\tables"
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe src\run_final_tuned_xgboost.py
.\venv\Scripts\python.exe src\run_calibration.py
.\venv\Scripts\python.exe src\write_reproduction_metadata.py
```

Run the threshold and reference-delay sensitivity cases, then combine those outputs with the 1 km/6 h baseline:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe src\run_sensitivity_experiments.py
.\venv\Scripts\python.exe src\aggregate_sensitivity_results.py
```

Sensitivity outputs are written under `results/experiments/sensitivity/`, including per-satellite CSVs, configuration rankings, a JSON manifest, and a summary plot. The run uses the existing 6-hour label set and filters it to narrower delay caps; for caps up to 6 hours this is equivalent to the label builder's nearest-reference rule. It rebuilds feature rows from raw TLEs and repeats the temporal split and LOSO configuration selection for each scenario.

All live Space-Track scripts require an account. Set `SPACE_TRACK_IDENTITY` and `SPACE_TRACK_PASSWORD` in the process environment before running the download, coverage, or verification scripts; do not place credentials in source files.

## Reproduced result snapshot

The temporal script selected its probability threshold using training data only and reproduced test ROC-AUC 0.9212, PR-AUC 0.6215, and F1 0.5892 on 2,595 test rows.

The final LOSO reproduction selected `strong_regularization`. Its unweighted mean across six satellite test folds was ROC-AUC 0.8360, PR-AUC 0.3872, Brier score 0.1128 raw / 0.1016 calibrated, and F1 0.2698 raw / 0.2452 calibrated. Per-fold results and calibration parameters are saved in `results/experiments/loso_reproduction/tables/`.

The separate temporal calibration experiment fits one Platt calibrator on all 2,011 validation rows, then applies it to temporal test and ISS test subsets. Its reproduced coefficient is 1.262472 and intercept is -0.720820 (`C=1e6`); these are stored with their fit protocol in `results/experiments/calibration_reproduction/tables/calibration_parameters.json`. This scalar pair belongs to that temporal calibration run; the final LOSO experiment has distinct per-fold pairs.

The requested sensitivity summary is `results/experiments/sensitivity/sensitivity_loso_summary.csv`; the complete per-fold table and protocol metadata are alongside it. Compare the sensitivity rows together with sample sizes and class balance before interpreting metric changes, especially for the 5 km target and the shorter reference-delay caps.

## Important source-data security note

The historical repository version contained a Space-Track credential in the live verification script. The current script reads credentials from environment variables, and no credential is needed for the checked-in data or offline experiments. Treat the old credential as exposed and rotate it before any further use.\n
