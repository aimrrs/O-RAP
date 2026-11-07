import math

import pytest

import app


def test_project_data_provides_all_satellite_samples_and_features():
    satellites = app.satellite_options()
    assert len(satellites) == 14
    assert len(satellites[0]["features"]) == 13
    assert all(math.isfinite(value) for value in satellites[0]["features"].values())


def test_research_page_uses_committed_test_results():
    data = app.research_data()
    assert data["dataset"] == {"pairs": 242547, "satellites": 14, "boundary_excluded": 28, "features": 13}
    assert data["test"]["samples"] == 36382
    assert data["test"]["positives"] == 3095
    assert data["test"]["pr_auc"] == pytest.approx(0.2990550442)


def test_tle_sample_is_original_source_and_parses_to_model_features():
    sample = app.satellite_options()[0]
    lines = sample["tle"].splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1 ") and lines[1].startswith("2 ")
    parsed = app.parse_tle(sample["tle"])
    assert set(parsed) == set(app.FEATURES)
    assert parsed["mean_motion"] == pytest.approx(sample["features"]["mean_motion"], abs=1e-8)
    assert parsed["eccentricity"] == pytest.approx(sample["features"]["eccentricity"], abs=1e-7)


def test_tle_parser_rejects_invalid_input():
    with pytest.raises(ValueError, match="TLE"):
        app.parse_tle("not a TLE")


def test_real_model_prediction_for_repository_record():
    sample = app.satellite_options()[0]
    result = app.predict(app.parse_tle(sample["tle"]))
    assert result["model"] == "orbital_only_xgboost + Platt scaling"
    assert 0 <= result["probability"] <= 1
    assert 0 < result["threshold"] < 1
    assert result["flagged"] == (result["probability"] >= result["threshold"])


def test_prediction_rejects_non_finite_feature():
    features = app.satellite_options()[0]["features"]
    features["eccentricity"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        app.predict(features)
