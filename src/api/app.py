from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import uuid

# Logging
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
    "surge_model_path": None,
    "loaded_at": None,
}

# Schemas
class HealthResponse(BaseModel):
    status: str
    demand_model_loaded: bool
    surge_model_loaded: bool

class ModelsResponse(BaseModel):
    loaded_at: str
    demand: Dict[str, Any]
    surge: Dict[str, Any]

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
    request_id: str
    predicted_demand: float
    model_used: str
    timestamp: str

class DemandBatchRequest(BaseModel):
    items: List[DemandPredictionRequest]

class DemandBatchResponse(BaseModel):
    request_id: str
    results: List[DemandPredictionResponse]
    timestamp: str

class SurgePredictionRequest(BaseModel):
    city: str = Field(..., example="San Francisco")
    zone_name: str = Field(..., example="Downtown")
    hour: int = Field(..., ge=0, le=23, example=18)
    day_of_week: int = Field(..., ge=1, le=7, example=5)
    is_weekend: bool = Field(..., example=False)

    predicted_demand: float = Field(..., example=60.0)
    supply_index: float = Field(..., example=0.85)  # < 1 means less supply
    rain_intensity: float = Field(..., example=0.2)

class SurgePredictionResponse(BaseModel):
    request_id: str
    predicted_surge_multiplier: float
    model_used: str
    timestamp: str

# Repo + paths
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _pred_log_base_dir() -> Path:
    return _repo_root() / "data" / "api_logs"

def _predictions_delta_path() -> str:
    return str(_repo_root() / "data" / "gold" / "api_predictions_delta")

def load_models() -> None:
    root = _repo_root()
    default_demand = root / "data" / "models" / "demand_forecasting" / "best_model.pkl"
    default_surge = root / "data" / "models" / "surge_pricing" / "best_model.pkl"
    demand_path = Path(os.getenv("DEMAND_MODEL_PATH", str(default_demand)))
    surge_path = Path(os.getenv("SURGE_MODEL_PATH", str(default_surge)))

    MODEL_META["demand_model_path"] = str(demand_path)
    MODEL_META["surge_model_path"] = str(surge_path)
    MODEL_META["loaded_at"] = datetime.utcnow().isoformat()

    # demand
    if demand_path.exists():
        try:
            import joblib
            MODELS["demand"] = joblib.load(demand_path)
            logger.info("Loaded demand model from %s", str(demand_path))
        except Exception:
            MODELS["demand"] = None
            logger.exception("Failed to load demand model from %s", str(demand_path))
    else:
        MODELS["demand"] = None
        logger.warning("Demand model not found at %s", str(demand_path))

    # surge
    if surge_path.exists():
        try:
            import joblib
            MODELS["surge"] = joblib.load(surge_path)
            logger.info("Loaded surge model from %s", str(surge_path))
        except Exception:
            MODELS["surge"] = None
            logger.exception("Failed to load surge model from %s", str(surge_path))
    else:
        MODELS["surge"] = None
        logger.warning("Surge model not found at %s", str(surge_path))

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
        df = pd.DataFrame([{
            "city": req.city,
            "zone_name": req.zone_name,
            "hour": req.hour,
            "day_of_week": req.day_of_week,
            "is_weekend": int(req.is_weekend),
            "demand_lag_1": req.demand_lag_1,
            "demand_avg_7d": req.demand_avg_7d,
            "avg_fare": req.avg_fare,
            "avg_surge": req.avg_surge,
        }])
        pred = model.predict(df)
        return float(pred[0])
    except Exception:
        logger.exception("Demand model prediction failed. Using fallback.")
        return _fallback_demand_prediction(req)

def _fallback_surge_prediction(req: SurgePredictionRequest) -> float:
    # Simple rule-based fallback: higher demand + low supply + rain => more surge
    demand_factor = min(max(req.predicted_demand / 60.0, 0.5), 2.0)
    supply_factor = 1.0 + max(1.0 - req.supply_index, 0.0) * 0.8
    rain_factor = 1.0 + min(max(req.rain_intensity, 0.0), 1.0) * 0.25
    raw = 1.0 * demand_factor * supply_factor * rain_factor
    return float(min(max(raw, 1.0), 3.0))

def _model_surge_prediction(req: SurgePredictionRequest) -> float:
    model = MODELS.get("surge")
    if model is None:
        return _fallback_surge_prediction(req)

    try:
        import pandas as pd
        df = pd.DataFrame([{
            "city": req.city,
            "zone_name": req.zone_name,
            "hour": req.hour,
            "day_of_week": req.day_of_week,
            "is_weekend": int(req.is_weekend),
            "predicted_demand": req.predicted_demand,
            "supply_index": req.supply_index,
            "rain_intensity": req.rain_intensity,
        }])
        pred = model.predict(df)
        value = float(pred[0])
        return float(min(max(value, 1.0), 3.0))
    except Exception:
        logger.exception("Surge model prediction failed. Using fallback.")
        return _fallback_surge_prediction(req)

# Background logging
def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _log_prediction_event(event: Dict[str, Any]) -> None:
    use_delta = os.getenv("API_LOG_TO_DELTA", "0") == "1"

    if use_delta:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.appName("mobility-api-logging").getOrCreate()
            df = spark.createDataFrame([event])
            df.write.format("delta").mode("append").save(_predictions_delta_path())
            return
        except Exception:
            logger.exception("Delta logging failed. Falling back to JSONL.")

    log_dir = _pred_log_base_dir()
    day = datetime.utcnow().strftime("%Y-%m-%d")
    _append_jsonl(log_dir / f"predictions_{day}.jsonl", event)

# Routes
@app.get("/")
def root():
    return {"message": "Distributed Mobility ML API is running"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        demand_model_loaded=MODELS.get("demand") is not None,
        surge_model_loaded=MODELS.get("surge") is not None,
    )

@app.get("/models", response_model=ModelsResponse)
def models_info():
    return ModelsResponse(
        loaded_at=str(MODEL_META.get("loaded_at") or ""),
        demand={
            "loaded": MODELS.get("demand") is not None,
            "path": MODEL_META.get("demand_model_path"),
        },
        surge={
            "loaded": MODELS.get("surge") is not None,
            "path": MODEL_META.get("surge_model_path"),
        },
    )

@app.post("/predict/demand", response_model=DemandPredictionResponse)
def predict_demand(request: DemandPredictionRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    started = datetime.utcnow()

    try:
        predicted = _model_demand_prediction(request)
        model_used = "local_artifact" if MODELS.get("demand") is not None else "fallback"

        response = DemandPredictionResponse(
            request_id=request_id,
            predicted_demand=round(float(predicted), 2),
            model_used=model_used,
            timestamp=datetime.utcnow().isoformat(),
        )

        latency_ms = int((datetime.utcnow() - started).total_seconds() * 1000)
        background_tasks.add_task(
            _log_prediction_event,
            {
                "request_id": request_id,
                "task": "demand",
                "model_used": model_used,
                "timestamp": response.timestamp,
                "latency_ms": latency_ms,
                "inputs": request.model_dump(),
                "outputs": {"predicted_demand": response.predicted_demand},
            },
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/demand/batch", response_model=DemandBatchResponse)
def predict_demand_batch(request: DemandBatchRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    started = datetime.utcnow()

    try:
        results: List[DemandPredictionResponse] = []
        for item in request.items:
            predicted = _model_demand_prediction(item)
            model_used = "local_artifact" if MODELS.get("demand") is not None else "fallback"
            results.append(
                DemandPredictionResponse(
                    request_id=request_id,
                    predicted_demand=round(float(predicted), 2),
                    model_used=model_used,
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        latency_ms = int((datetime.utcnow() - started).total_seconds() * 1000)
        background_tasks.add_task(
            _log_prediction_event,
            {
                "request_id": request_id,
                "task": "demand_batch",
                "model_used": "local_artifact" if MODELS.get("demand") is not None else "fallback",
                "timestamp": datetime.utcnow().isoformat(),
                "latency_ms": latency_ms,
                "batch_size": len(request.items),
            },
        )

        return DemandBatchResponse(
            request_id=request_id,
            results=results,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/surge", response_model=SurgePredictionResponse)
def predict_surge(request: SurgePredictionRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    started = datetime.utcnow()

    try:
        predicted = _model_surge_prediction(request)
        model_used = "local_artifact" if MODELS.get("surge") is not None else "fallback"

        response = SurgePredictionResponse(
            request_id=request_id,
            predicted_surge_multiplier=round(float(predicted), 3),
            model_used=model_used,
            timestamp=datetime.utcnow().isoformat(),
        )

        latency_ms = int((datetime.utcnow() - started).total_seconds() * 1000)

        background_tasks.add_task(
            _log_prediction_event,
            {
                "request_id": request_id,
                "task": "surge",
                "model_used": model_used,
                "timestamp": response.timestamp,
                "latency_ms": latency_ms,
                "inputs": request.model_dump(),
                "outputs": {"predicted_surge_multiplier": response.predicted_surge_multiplier},
            },
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")