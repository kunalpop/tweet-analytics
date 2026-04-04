"""
Retweet Prediction API - Production Ready for Render.com
With Inference Logging for Monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import logging

import re

from torch import embedding

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

prediction_logger = logging.getLogger("predictions")
prediction_handler = logging.FileHandler(LOG_DIR / "predictions.jsonl")
prediction_handler.setFormatter(logging.Formatter('%(message)s'))
prediction_logger.addHandler(prediction_handler)
prediction_logger.setLevel(logging.INFO)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Virality Prediction API",
    description="Ensemble retweet count prediction using XGBoost + Random Forest Pipelines",
    version="1.0.0"
)

# ============================================================
# LOAD PIPELINES, EMBEDDER & METADATA
# ============================================================
MODEL_DIR = Path("models")

try:
    gb_pipeline = joblib.load(MODEL_DIR / "gb_pipeline_model.pkl")
    rf_pipeline = joblib.load(MODEL_DIR / "rf_pipeline_model.pkl")
    embedder    = joblib.load(MODEL_DIR / "embedder.pkl")

    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    with open(MODEL_DIR / "feature_stats.json") as f:
        feature_stats = json.load(f)

    # Embedding drift stats loaded from feature_stats.json
    embedding_mean = np.array(feature_stats["embedding_mean"])  # shape: (embedding_dim,)
    embedding_std  = np.array(feature_stats["embedding_std"])   # shape: (embedding_dim,)

    logger.info("Models and embedder loaded successfully.")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class TweetInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="Raw tweet text (up to 280 characters)",
        examples=["Just launched our new product! Check it out 🚀"]
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Tweet text cannot be empty or whitespace.')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Just launched our new product! Check it out 🚀"
            }
        }


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    xgboost_prediction: float
    random_forest_prediction: float
    ensemble_prediction: float
    drift_warnings: List[str]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def embed(text: str) -> np.ndarray:
    """Embed tweet text into a feature vector."""
    embedding = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].reshape(1, -1)     # shape: (1, embedding_dim)
    return embedding                                                                                                # shape: (embedding_dim,)


def check_embedding_drift(embedding: np.ndarray) -> list:
    """
    Check if the embedding is an outlier relative to training embeddings.
    Flags any dimension where the value exceeds mean ± 3*std, using
    per-dimension mean and std vectors from training.

    Supports embedding shape (1, embedding_dim) or (embedding_dim,).
    """
    warnings = []

    # Flatten if input is 2D (1, embedding_dim)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding.flatten()  # shape: (embedding_dim,)

    # Calculate deviations
    deviations   = np.abs(embedding - embedding_mean)   # shape: (embedding_dim,)
    threshold    = 3.0 * embedding_std                 # shape: (embedding_dim,)
    drifted_dims = np.where(deviations > threshold)[0] # indices of flagged dims

    if len(drifted_dims) > 0:
        worst_idx = drifted_dims[np.argmax(deviations[drifted_dims])]
        warnings.append(
            f"Embedding drift detected: {len(drifted_dims)} dimension(s) exceed 3σ. "
            f"Worst — dim {worst_idx}: value={embedding[worst_idx]:.4f}, "
            f"mean={embedding_mean[worst_idx]:.4f}, "
            f"threshold=±{threshold[worst_idx]:.4f}"
        )

    return warnings

# ============================================================
# GLOBAL STATE
# ============================================================
request_counter = 0


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """Health check."""
    return {
        "status": "healthy",
        "message": "Retweet Prediction API is running",
        "models": ["XGBoost", "Random Forest"],
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": True,
        "total_predictions": request_counter
    }


@app.get("/model-info")
def model_info():
    """Get model metadata."""
    return {
        "input"            : "raw tweet text (embedded internally)",
        "embedding_dim"    : feature_stats["embedding_dim"],
        "pipeline_steps"   : metadata.get("pipeline_steps", []),
        "training_samples" : metadata.get("training_samples"),
        "test_samples"     : metadata.get("test_samples"),
    }


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)            # remove URLs
    text = re.sub(r'@\w+', '', text)                       # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)                  # strip # but keep word
    text = re.sub(r'^rt\s+', '', text, flags=re.MULTILINE) # remove RT prefix
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)            # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()               # collapse whitespace
    return text

@app.post("/predict", response_model=PredictionResponse)
def predict(tweet: TweetInput):
    """Predict retweet count for a tweet."""
    global request_counter
    request_counter += 1
    prediction_id = f"twt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_counter:06d}"

    try:
        # Embed the tweet text
        cleaned_text = clean_text(tweet.text)
        X = embed(cleaned_text)           # shape: (embedding_dim,)

        print(X)

        # Drift detection on embedding
        drift_warnings = check_embedding_drift(X)

        # Predict retweet counts
        gb_pred = gb_pipeline.predict(X)[0]
        rf_pred  = rf_pipeline.predict(X)[0]
        ensemble_pred = (gb_pred + rf_pred) / 2

        # Clamp to non-negative (retweet counts can't be negative)
        gb_pred      = int(max(0.0, np.expm1(gb_pred)))
        rf_pred       = int(max(0.0, np.expm1(rf_pred)))
        ensemble_pred = int(max(0.0, np.expm1(ensemble_pred)))

        # Log prediction
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_id": prediction_id,
            "input_text": tweet.text,
            "predictions": {
                "xgboost":       {"retweets": gb_pred},
                "random_forest": {"retweets": rf_pred},
                "ensemble":      {"retweets": ensemble_pred},
            },
            "has_drift": len(drift_warnings) > 0,
            "drift_warnings": drift_warnings,
        }
        prediction_logger.info(json.dumps(log_entry))
        logger.info(f"[{prediction_id}] Predicted retweets: {ensemble_pred:.1f}")

        return PredictionResponse(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            xgboost_prediction=round(gb_pred, 2),
            random_forest_prediction=round(rf_pred, 2),
            ensemble_prediction=round(ensemble_pred, 2),
            drift_warnings=drift_warnings,
        )

    except Exception as e:
        logger.error(f"[{prediction_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary")
def get_log_summary():
    """Get summary of logged predictions for monitoring."""
    log_file = LOG_DIR / "predictions.jsonl"

    if not log_file.exists():
        return {
            "total_predictions": 0,
            "avg_predicted_retweets": 0,
            "predictions_with_drift": 0,
            "drift_rate": 0,
            "message": "No predictions logged yet"
        }

    total        = 0
    retweet_sum  = 0.0
    drift_count  = 0

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total       += 1
                retweet_sum += entry["predictions"]["ensemble"]["retweets"]
                if entry.get("has_drift", False):
                    drift_count += 1
            except:
                continue

    return {
        "total_predictions":       total,
        "avg_predicted_retweets":  round(retweet_sum / total, 2) if total > 0 else 0,
        "predictions_with_drift":  drift_count,
        "drift_rate":              round(drift_count / total * 100, 2) if total > 0 else 0,
    }