from datetime import datetime, timezone

from predict import predict

CASES = [
    ("ISS", "1 25544U 98067A   26255.59495341  .00005107  00000+0  10048-3 0  9998", "2 25544  51.6306 227.4903 0004922 132.5649 227.5755 15.49091282585302"),
    ("Hubble", "1 20580U 90037B   25270.00000000  .00000800  00000+0  00000+0 0  9999", "2 20580  28.4690 100.0000 0002800 100.0000 260.0000 15.09200000123456"),
    ("NOAA 19", "1 33591U 09005A   25270.00000000  .00000100  00000+0  00000+0 0  9999", "2 33591  99.1900 200.0000 0014000 100.0000 260.0000 14.12000000123456"),
    ("Sentinel-1A", "1 39634U 14016A   25270.00000000  .00000500  00000+0  00000+0 0  9999", "2 39634  98.1800 150.0000 0001200 100.0000 260.0000 14.59000000123456"),
    ("TDRS-5", "1 24876U 97034A   25270.00000000  .00000010  00000+0  00000+0 0  9999", "2 24876   4.0000 250.0000 0005000 100.0000 260.0000  1.00270000123456"),
    ("GPS BIIR-2", "1 26407U 00040A   24366.82702509  .00000075  00000-0  00000-0 0  9999", "2 26407  54.9915 236.3213 0137350 297.9239  69.6392  2.00563101179273"),
]


def main():
    total = 0
    for name, line1, line2 in CASES:
        # Each case uses an aware creation time after its TLE epoch.
        year = 2026 if line1[18:20] == "26" else (2000 + int(line1[18:20]) if int(line1[18:20]) < 57 else 1900 + int(line1[18:20]))
        creation = datetime(year + 1, 1, 2, tzinfo=timezone.utc)
        for horizon in (6, 12, 24, 48):
            result = predict(line1, line2, creation, horizon)
            assert result["sgp4"]["error_code"] == 0, name
            assert result["sgp4"]["frame"] == "TEME"
            assert 0.0 <= result["model_score"] <= 1.0
            assert result["horizon_hours"] == horizon
            assert result["target_time"] == (creation.replace() + __import__("datetime").timedelta(hours=horizon)).isoformat()
            total += 1
    print(f"Inference smoke check passed: {total} satellite-horizon cases.")


if __name__ == "__main__":
    main()
