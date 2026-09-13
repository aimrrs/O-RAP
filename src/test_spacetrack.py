import requests
from datetime import datetime, timezone

USERNAME = "aimrrs404@gmail.com"
PASSWORD = "qwertyuioplkjhgfdsa"

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
HISTORY_URL = (
    "https://www.space-track.org/basicspacedata/query/"
    "class/gp_history/"
    "NORAD_CAT_ID/25544/"
    "EPOCH/%3E2025-01-01/"
    "orderby/CREATION_DATE%20asc/"
    "limit/100/"
    "format/json"
)

session = requests.Session()

login_data = {
    "identity": USERNAME,
    "password": PASSWORD,
}

login_response = session.post(LOGIN_URL, data=login_data)
print("Login status:", login_response.status_code)

response = session.get(HISTORY_URL)
print("History status:", response.status_code)

records = response.json()

print("Records:", len(records))
print()

previous = None

for r in records:
    creation = datetime.fromisoformat(
        r["CREATION_DATE"].replace("Z", "+00:00")
    )

    epoch = datetime.fromisoformat(
        r["EPOCH"].replace("Z", "+00:00")
    )

    print(
        f"CREATION: {creation} | "
        f"EPOCH: {epoch} | "
        f"EPOCH-AGE: {(creation - epoch).total_seconds() / 3600:.2f} h"
    )

    if previous is not None:
        delta = (creation - previous).total_seconds() / 3600
        print(f"  Time since previous TLE: {delta:.2f} h")

    previous = creation