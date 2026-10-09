# O-RAP interactive research prototype

This folder adds two local web pages to the O-RAP research repository:

- `/` — TLE input, SGP4 propagation, and an exploratory model score.
- `/experiments` — methods, paper-aligned result summaries, plots, interpretation, and limitations.

The demo is a research showcase. Its bundled XGBoost artifact is separate from the paper-aligned evaluation artifacts in `results/experiments/paper_final/`. It reports an **uncalibrated model score**, not a calibrated probability or an operational risk decision. Its target is framed as later-TLE agreement, not independently measured orbital truth.

## Run locally

From the repository root in PowerShell:

```powershell
py -3.14 -m venv .webapp-venv
.\.webapp-venv\Scripts\python.exe -m pip install -r webapp\requirements.txt
.\.webapp-venv\Scripts\python.exe -m uvicorn webapp.app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/experiments`. The app binds to localhost by default and does not require deployment or credentials.

## Demo behavior

The input is a standard 69-character TLE pair, the UTC TLE creation timestamp, and a supported horizon (6, 12, 24, or 48 hours). Source TLE age is reconstructed as creation timestamp minus TLE epoch. SGP4 propagates the source TLE to creation time plus the selected horizon and returns position/velocity in TEME. The page makes no TEME-to-ITRF conversion.

The bundled model metadata describes an artifact trained for January–August 2025 and calibrated on September–October. Its saved calibration JSON is retained as provenance but is not applied by the demo because transfer calibration did not improve the paper-aligned nested diagnostic. The demo artifact is not the exact model run behind the manuscript's aggregate metrics. The optional `src/train_production_model.py` helper expects separate `webapp/data/train_temporal.csv` and `webapp/data/validation_temporal.csv` inputs, which are not included; those files should not be recreated from test rows.

## Experiments and evidence

The experiments page summarizes the result snapshots and interpretation. The source CSV/JSON outputs remain in `../results/experiments/paper_final/`; legacy sensitivity outputs remain in `../results/experiments/sensitivity/`. See the repository root README for exact reproduction commands and data provenance, licensing, and redistribution caveats.

The results cover six satellites from 2025. The label is a later-TLE SGP4 agreement proxy. Earlier development examined November–December, so the chronological test is not prospective untouched validation. Do not present demo scores or results as operationally validated risk estimates.

## Smoke checks

With the app dependencies installed, run from the repository root:

```powershell
.\.webapp-venv\Scripts\python.exe webapp\src\test_features.py
.\.webapp-venv\Scripts\python.exe webapp\src\test_model_artifact.py
.\.webapp-venv\Scripts\python.exe webapp\src\test_multisatellite_inference.py
```
