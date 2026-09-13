"""
O-RAP
Orbital Reliability Assessment and Prediction

Phase 1: Data Feasibility Pilot
TLE download + SGP4 propagation test.
"""

from urllib.request import urlopen
from sgp4.api import Satrec, jday
from datetime import datetime, timezone


TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"


def download_tle():
    with urlopen(TLE_URL, timeout=20) as response:
        lines = response.read().decode("utf-8").strip().splitlines()

    return lines[0], lines[1], lines[2]


def propagate_tle(name, line1, line2, dt):
    satellite = Satrec.twoline2rv(line1, line2)

    jd, fr = jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond / 1_000_000,
    )

    error, position, velocity = satellite.sgp4(jd, fr)

    return error, position, velocity


if __name__ == "__main__":
    name, line1, line2 = download_tle()

    now = datetime.now(timezone.utc)

    error, position, velocity = propagate_tle(
        name,
        line1,
        line2,
        now,
    )

    print("Satellite:", name)
    print("Propagation time:", now.isoformat())
    print()
    print("SGP4 error code:", error)
    print("Position (km):", position)
    print("Velocity (km/s):", velocity)

    if error == 0:
        print()
        print("✓ SGP4 propagation successful.")
    else:
        print()
        print("✗ SGP4 propagation returned an error.")