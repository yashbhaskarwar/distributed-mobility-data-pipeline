from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict


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
MODELS: Dict[str, object] = {}

# Schemas
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

# Model loading stub
def load_models() -> None:
    # Placeholder 
    MODELS["demand"] = None
    MODELS["surge"] = None
    logger.info("Model registry initialized")

# Startup event
@app.on_event("startup")
def startup_event():
    logger.info("Starting Mobility ML API")
    load_models()

# Routes
@app.get("/")
def root():
    return {"message": "Distributed Mobility ML API is running"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    models_loaded = all(model is not None for model in MODELS.values())
    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
    )