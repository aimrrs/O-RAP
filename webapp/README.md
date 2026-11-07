# O-RAP research interface

The interface has three pages: Prediction, Research, and About. It uses the repository's saved orbital-only XGBoost model and Platt calibrator, and reads research summaries from the committed output files. It does not retrain or modify the research pipeline.

## Run locally

From the repository root, create the app-local environment, install the project dependencies, and start the server:

```powershell
python -m venv webapp/.venv
webapp/.venv/Scripts/python.exe -m pip install -r requirements.txt
webapp/.venv/Scripts/python.exe webapp/app.py
```

Open <http://127.0.0.1:8000>. To use another local port, pass it as an argument, for example `python webapp/app.py 8080`.

The Prediction page accepts a standard two-line TLE. You can paste your own set or load an example using the original TLE lines stored in the repository's historical GP records. O-RAP parses the same 13 orbital features used by the orbital-only research model, then runs the committed estimator and Platt calibrator. The flag uses the F1 threshold selected on the validation partition and reported in `results/modeling/experiments/orbital_only_analysis/orbital_only_analysis_configuration.json`.

Navigation uses clean paths: `/`, `/research`, and `/about`. There is no hash-based page routing.

## Interpretation

The model estimates the chance that a source record is associated with more than 1 km of disagreement between successive TLE-based SGP4 position estimates in the study's comparison. This is a research result, not a measure of independently observed physical orbit error, collision risk, or operational safety.
