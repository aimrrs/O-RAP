from datetime import datetime
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_DIR = Path(__file__).resolve().parents[2]
APP_DIR = REPO_DIR / "webapp" / "app"
SRC_DIR = REPO_DIR / "webapp" / "src"
STATIC_DIR = APP_DIR / "static"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import ALLOWED_HORIZONS, predict

app = FastAPI(
    title="O-RAP",
    description="Interactive research prototype for TLE-derived SGP4 error-proxy scoring.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    tle_line1: str = Field(..., min_length=69, max_length=69)
    tle_line2: str = Field(..., min_length=69, max_length=69)
    creation_time: datetime
    horizon_hours: float


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/experiments", include_in_schema=False)
def experiments_page():
    return FileResponse(STATIC_DIR / "experiments.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def run_prediction(request: PredictionRequest):
    if request.creation_time.tzinfo is None:
        raise HTTPException(status_code=400, detail="creation_time must include a timezone.")
    if request.horizon_hours not in ALLOWED_HORIZONS:
        raise HTTPException(status_code=400, detail="horizon_hours must be one of 6, 12, 24, or 48.")
    try:
        return predict(
            line1=request.tle_line1,
            line2=request.tle_line2,
            creation_time=request.creation_time,
            horizon_hours=request.horizon_hours,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
