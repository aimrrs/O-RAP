# O-RAP v2.0

## Orbital Reliability Assessment and Prediction

O-RAP is a machine-learning research framework that investigates the reliability of SGP4-based orbital predictions using historical Two-Line Element (TLE) data.

The v2.0 research evaluates whether TLE-derived orbital characteristics can help estimate the risk of significant position disagreement between consecutive TLE records.

## Research Scope

- **Satellites:** 14 satellites
- **Input data:** Historical General Perturbations (GP) element records in JSON format
- **Orbit propagation:** SGP4
- **Reference frame:** TEME
- **Primary disagreement threshold:** 1 km
- **Machine learning:** XGBoost and Logistic Regression
- **Evaluation:** Chronological holdout and Leave-One-Satellite-Out (LOSO)
- **Probability calibration:** Platt scaling

## Methodology

The framework constructs pairs from consecutive distinct TLE epochs for each satellite.

For each pair, the older TLE is propagated to the newer TLE's epoch. Its predicted position is compared with the position obtained by propagating the newer TLE at that same epoch.

The resulting three-dimensional position disagreement is used to construct a binary reliability label:

- **Reliable (0):** Position disagreement is at or below 1 km.
- **Unreliable (1):** Position disagreement exceeds 1 km.

**Scientific qualification:** These labels represent disagreement between two TLE-derived SGP4 position estimates. They are not independently measured physical orbit errors.

## Dataset

- 14 satellites
- 242,575 consecutive TLE pairs
- 19,827 positive labels
- 222,748 negative labels
- 13 orbital features and one retrospective interval feature

The dataset is partitioned chronologically using unique TLE epochs per satellite:

- Training: earliest 70%
- Validation: next 15%
- Testing: latest 15%

Pairs crossing partition boundaries are excluded to prevent shared TLE epochs between splits.

## Machine Learning and Evaluation

The research evaluates:

- XGBoost classification
- Logistic Regression baseline
- Orbital-only and interval-inclusive feature configurations
- Chronological test performance
- Leave-One-Satellite-Out generalization
- Probability calibration
- Threshold sensitivity

The orbital-only configuration excludes the interval feature because the interval between consecutive TLE records is known only after the newer record becomes available. Therefore, interval-inclusive experiments are retrospective comparisons rather than strictly pre-use predictions.

## Repository Structure

- `data/raw/` - Original satellite GP history records
- `data/processed/` - Processed consecutive TLE pairs
- `src/` - Data processing, validation, modeling and analysis scripts
- `results/consecutive_tle_pipeline/` - Pipeline outputs
- `results/modeling/` - Model experiments and evaluation results
- `results/consecutive_sgp4_validation/` - SGP4 validation
- `results/consecutive_pair_audit/` - Pair construction audit
- `results/consecutive_pair_interval_audit/` - Interval audit
- `results/data_audit/` - Raw data audit
- `results/tle_forensic_audit/` - Forensic analysis
- `results/figures/` - Research figures
- `results/final_tables/` - Final research tables
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies

## Reproducibility

The repository includes:

- Raw TLE-derived datasets
- Consecutive-pair construction scripts
- SGP4 validation and audit scripts
- Feature extraction and chronological splitting
- Model training and evaluation scripts
- LOSO and threshold sensitivity experiments
- Generated research figures and result tables

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

Run the relevant scripts from the repository root.

## Project Status

O-RAP v2.0 contains the expanded 14-satellite consecutive-TLE research workflow, model evaluation, validation evidence, and publication figures and tables.

The research is exploratory. Its position-disagreement labels should not be interpreted as ground-truth physical orbit prediction errors.