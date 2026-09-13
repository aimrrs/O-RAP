# O-RAP — Orbital Reliability Assessment and Prediction

Research repository for **O-RAP**, a lightweight framework for estimating the reliability of TLE/SGP4 orbit predictions under satellite and temporal distribution shift.

## Research Question

Can a lightweight, calibrated model predict the probability that an SGP4 prediction will exceed an application-defined error tolerance, including for satellites and time periods unseen during training?

## Approach

- Historical TLE data from Space-Track
- SGP4-based orbit propagation
- TLE-to-TLE proxy error labels
- 1 km reliability threshold
- XGBoost-based reliability estimation
- Probability calibration
- Future-temporal and unseen-satellite evaluation
- Distribution-shift analysis

## Dataset

Six satellites were evaluated across 2025:

- ISS
- Hubble Space Telescope
- NOAA 19
- Sentinel-1A
- TDRS-5
- GPS BIIR-2

The final dataset contains **17,263 evaluation samples**.

## Repository Contents

```text
data/          Research datasets and processed data
src/           Data processing and experiment scripts
results/       Experimental results, tables, and figures
experiments/   Experimental work
paper/         Research paper materials
```

## Note

The reference orbit used for labeling is an independently published later TLE propagated with SGP4 and is therefore treated as a **proxy reference**, not physical ground truth.

This repository contains the research and experimental work behind O-RAP.
