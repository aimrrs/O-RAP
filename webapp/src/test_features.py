from datetime import datetime, timezone

from features import FEATURE_NAMES, build_features, calculate_tle_age_hours

TLE_LINE1 = "1 25544U 98067A   26255.59495341  .00005107  00000+0  10048-3 0  9998"
TLE_LINE2 = "2 25544  51.6306 227.4903 0004922 132.5649 227.5755 15.49091282585302"


def main():
    created = datetime(2026, 9, 13, tzinfo=timezone.utc)
    features = build_features(TLE_LINE1, TLE_LINE2, 24, created)
    assert list(features) == FEATURE_NAMES
    assert abs(features["source_tle_age_hours"] - calculate_tle_age_hours(TLE_LINE1, created)) < 1e-9
    assert features["source_tle_age_hours"] > 0
    assert features["horizon_hours"] == 24
    assert all(isinstance(value, float) for value in features.values())
    print("Feature order and creation-minus-TLE-epoch semantics verified.")


if __name__ == "__main__":
    main()
