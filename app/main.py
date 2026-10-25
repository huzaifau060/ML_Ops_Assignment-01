import json
import logging
import os
import sys
from typing import Dict, Optional
import joblib
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, create_model, Field

# Configure logging to stdout
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("triage_service")

# Environment configurations
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
REDIS_HOST = os.getenv("REDIS_HOST", None)
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Feature list
FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error",
    "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]

# Create Pydantic model dynamically requiring all 30 features >= 0
field_definitions = {name: (float, Field(..., ge=0.0)) for name in FEATURE_NAMES}
PredictionInput = create_model("PredictionInput", **field_definitions)


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    label: str
    malignant_probability: float
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


app = FastAPI(title="Breast Cancer Triage Service", version=MODEL_VERSION)

# Global model and metadata state
model = None
model_card = {}
redis_client = None


def get_redis_client():
    global redis_client
    if REDIS_HOST and redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                socket_connect_timeout=2.0,
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Redis client: {e}")
    return redis_client


@app.on_event("startup")
def startup_event():
    global model, model_card
    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"FATAL: Model artifact not found at {MODEL_PATH}")
        sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model artifact loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model artifact: {e}")
        sys.exit(1)

    card_path = os.path.join(os.path.dirname(MODEL_PATH), "model_card.json")
    if os.path.exists(card_path):
        try:
            with open(card_path, "r") as f:
                model_card = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read model_card.json: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    # Prepare features in the exact ordered sequence
    payload_dict = payload.dict()
    feature_values = [payload_dict[name] for name in FEATURE_NAMES]
    import pandas as pd
    input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    try:
        # In breast cancer dataset: class 0 is malignant
        # Model returns probability for each class [prob_0, prob_1]
        probs = model.predict_proba(input_df)[0]
        malignant_prob = float(probs[0])
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Re-read dynamic threshold if set
    current_threshold = float(os.getenv("DECISION_THRESHOLD", str(DECISION_THRESHOLD)))
    label = "malignant" if malignant_prob >= current_threshold else "benign"

    # Increment Redis counter if REDIS_HOST is set
    client = get_redis_client()
    if client:
        try:
            client.incr("prediction_count")
        except Exception as e:
            logger.warning(f"Redis increment failed: {e}")

    return {
        "label": label,
        "malignant_probability": round(malignant_prob, 4),
        "threshold": current_threshold,
        "model_version": MODEL_VERSION,
    }


@app.get("/model/info")
def model_info():
    import sklearn
    test_metrics = model_card.get("test_metrics", {})
    return {
        "model_version": MODEL_VERSION,
        "model_family": model_card.get("model_family", "LogisticRegression"),
        "scikit_learn_version": sklearn.__version__,
        "git_commit": GIT_SHA,
        "decision_threshold": float(os.getenv("DECISION_THRESHOLD", str(DECISION_THRESHOLD))),
        "feature_count": len(FEATURE_NAMES),
        "test_set_metrics": test_metrics,
    }


@app.get("/stats")
def stats():
    client = get_redis_client()
    if client is None:
        return {"predictions_served": 0, "redis_connected": False}
    try:
        count = client.get("prediction_count")
        count_val = int(count) if count is not None else 0
        return {"predictions_served": count_val, "redis_connected": True}
    except Exception as e:
        logger.warning(f"Failed to fetch stats from Redis: {e}")
        return {"predictions_served": 0, "redis_connected": False, "error": str(e)}
