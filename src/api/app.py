from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mobility-api")

# FastAPI app
app = FastAPI(
    title="Distributed Mobility ML API",
    description="Real-time inference service",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model registry 
MODELS: Dict[str, Any] = {
    "demand": None,
    "surge": None,  
}

MODEL_META: Dict[str, Any] = {
    "demand_model_path": None,
    "loaded_at": None,
}

# Schemas
class HealthResponse(BaseModel):
    status: str
    demand_model_loaded: bool

class DemandPredictionRequest(BaseModel):
    city: str = Field(..., example="San Francisco")
    zone_name: str = Field(..., example="Downtown")
    hour: int = Field(..., ge=0, le=23, example=8)
    day_of_week: int = Field(..., ge=1, le=7, example=3)
    is_weekend: bool = Field(..., example=False)

    demand_lag_1: float = Field(..., example=45.0)
    demand_avg_7d: float = Field(..., example=43.5)

    avg_fare: float = Field(..., example=15.5)
    avg_surge: float = Field(..., example=1.2)

class DemandPredictionResponse(BaseModel):
    predicted_demand: float
    model_used: str
    timestamp: str

# Model loading 
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_models() -> None:
    root = _repo_root()
    default_path = root / "data" / "models" / "demand_forecasting" / "best_model.pkl"
    demand_model_path = Path(os.getenv("DEMAND_MODEL_PATH", str(default_path)))

    MODEL_META["demand_model_path"] = str(demand_model_path)
    MODEL_META["loaded_at"] = datetime.utcnow().isoformat()

    if not demand_model_path.exists():
        MODELS["demand"] = None
        logger.warning("Demand model not found at %s. Using fallback predictions.", str(demand_model_path))
        return

    try:
        import joblib

        MODELS["demand"] = joblib.load(demand_model_path)
        logger.info("Loaded demand model from %s", str(demand_model_path))
    except Exception:
        MODELS["demand"] = None
        logger.exception("Failed to load demand model from %s. Using fallback predictions.", str(demand_model_path))

@app.on_event("startup")
def startup_event():
    logger.info("Starting Mobility ML API")
    load_models()

def _fallback_demand_prediction(req: DemandPredictionRequest) -> float:
    base = req.demand_avg_7d
    peak = 1.25 if req.hour in (7, 8, 9, 16, 17, 18) else 1.0
    weekend = 0.85 if req.is_weekend else 1.0
    surge = 1.0 + max(req.avg_surge - 1.0, 0.0) * 0.15
    return float(base * peak * weekend * surge)

def _model_demand_prediction(req: DemandPredictionRequest) -> float:
    model = MODELS.get("demand")
    if model is None:
        return _fallback_demand_prediction(req)
    try:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "city": req.city,
                    "zone_name": req.zone_name,
                    "hour": req.hour,
                    "day_of_week": req.day_of_week,
                    "is_weekend": int(req.is_weekend),
                    "demand_lag_1": req.demand_lag_1,
                    "demand_avg_7d": req.demand_avg_7d,
                    "avg_fare": req.avg_fare,
                    "avg_surge": req.avg_surge,
                }
            ]
        )

        pred = model.predict(df)
        return float(pred[0])
    except Exception:
        logger.exception("Demand model prediction failed. Using fallback.")
        return _fallback_demand_prediction(req)

# Routes
@app.get("/")
def root():
    return {"message": "Distributed Mobility ML API is running"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        demand_model_loaded=MODELS.get("demand") is not None,
    )

@app.post("/predict/demand", response_model=DemandPredictionResponse)
def predict_demand(request: DemandPredictionRequest):
    try:
        predicted = _model_demand_prediction(request)
        model_used = "local_artifact" if MODELS.get("demand") is not None else "fallback"

        return DemandPredictionResponse(
            predicted_demand=round(float(predicted), 2),
            model_used=model_used,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")