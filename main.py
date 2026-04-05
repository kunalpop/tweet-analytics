"""
Virality Prediction API - Hybrid Ensemble (Embedding + NMF)
Production Ready with Monitoring + Observability
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import joblib
import json
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import logging
import re

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
    description="Hybrid ensemble (GB + RF + NMF) for predicting retweets and likes",
    version="2.0.0"
)

# ============================================================
# LOAD MODELS
# ============================================================
MODEL_DIR = Path("models")

try:
    gb_pipeline = joblib.load(MODEL_DIR / "gb_pipeline_model.pkl")
    rf_pipeline = joblib.load(MODEL_DIR / "rf_pipeline_model.pkl")
    rt_pipeline = joblib.load(MODEL_DIR / "retweet_pipeline.pkl")
    lk_pipeline = joblib.load(MODEL_DIR / "likes_pipeline.pkl")
    embedder    = joblib.load(MODEL_DIR / "embedder.pkl")

    with open(MODEL_DIR / "feature_stats.json") as f:
        feature_stats = json.load(f)

    embedding_mean = np.array(feature_stats["embedding_mean"])
    embedding_std  = np.array(feature_stats["embedding_std"])

    logger.info("All models loaded successfully.")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class TweetInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=280)

    @field_validator('text')
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Tweet text cannot be empty.")
        return v


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    xgboost_prediction: float
    random_forest_prediction: float
    nmf_prediction: float
    ensemble_prediction: float
    predicted_likes: float
    drift_warnings: List[str]

# ============================================================
# HELPERS
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'^rt\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def embed(text):
    return embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)


def check_embedding_drift(embedding):
    embedding = embedding.flatten()
    deviations = np.abs(embedding - embedding_mean)
    threshold = 3 * embedding_std

    drifted = np.where(deviations > threshold)[0]

    if len(drifted) > 0:
        return [f"{len(drifted)} embedding dims drifted"]

    return []

# ============================================================
# GLOBAL STATE
# ============================================================
request_counter = 0

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "status": "healthy",
        "message": "Virality Prediction API is running",
        "models": ["XGBoost", "Random Forest", "NMF + Linear"],
    }

# Ignore socket.io 404 polling logs from Jupyter/WebSockets
@app.get("/ws/socket.io/")
def ignore_socket_io():
    return {"message": "Ignored"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "total_predictions": request_counter
    }


@app.get("/model-info")
def model_info():
    return {
        "input": "raw tweet text",
        "models": {
            "retweet_models": [
                "Gradient Boosting (embedding-based)",
                "Random Forest (embedding-based)",
                "NMF + Linear Regression (text-based)"
            ],
            "likes_model": "NMF + Linear Regression"
        },
        "ensemble_method": "simple average of 3 models",
        "embedding_dim": len(embedding_mean),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tweet: TweetInput):
    global request_counter
    request_counter += 1

    prediction_id = f"twt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_counter:06d}"

    try:
        text = clean_text(tweet.text)

        # =========================
        # Embedding-based models
        # =========================
        X_embed = embed(text)

        gb_pred = gb_pipeline.predict(X_embed)[0]
        rf_pred = rf_pipeline.predict(X_embed)[0]

        # =========================
        # NMF-based pipelines
        # =========================
        nmf_pred = rt_pipeline.predict([text])[0]
        lk_pred  = lk_pipeline.predict([text])[0]

              # =========================
        # Post-processing
        # =========================
        gb_pred       = int(max(0, np.expm1(gb_pred)))
        rf_pred       = int(max(0, np.expm1(rf_pred)))
        nmf_pred = int(max(0, nmf_pred))
        lk_pred  = int(max(0, lk_pred))

        # =========================
        # Ensemble
        # =========================
        ensemble_pred = int((gb_pred + rf_pred + nmf_pred) / 3)

        drift_warnings = check_embedding_drift(X_embed)

        # =========================
        # Logging
        # =========================
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_id": prediction_id,
            "text": tweet.text,
            "retweets": ensemble_pred,
            "likes": lk_pred,
            "drift": drift_warnings
        }
        prediction_logger.info(json.dumps(log_entry))
        logger.info(f"[{prediction_id}] Retweets={ensemble_pred}, Likes={lk_pred}")

        return PredictionResponse(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            xgboost_prediction=gb_pred,
            random_forest_prediction=rf_pred,
            nmf_prediction=nmf_pred,
            ensemble_prediction=ensemble_pred,
            predicted_likes=lk_pred,
            drift_warnings=drift_warnings,
        )

    except Exception as e:
        logger.error(f"[{prediction_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary")
def get_log_summary():
    log_file = LOG_DIR / "predictions.jsonl"

    if not log_file.exists():
        return {
            "total_predictions": 0,
            "avg_retweets": 0,
            "avg_likes": 0,
            "drift_rate": 0,
            "message": "No logs yet"
        }

    total = 0
    retweet_sum = 0
    likes_sum = 0
    drift_count = 0

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                retweet_sum += entry.get("retweets", 0)
                likes_sum += entry.get("likes", 0)
                if entry.get("drift"):
                    drift_count += 1
            except:
                continue

    return {
        "total_predictions": total,
        "avg_retweets": round(retweet_sum / total, 0) if total else 0,
        "avg_likes": round(likes_sum / total, 0) if total else 0,
        "drift_rate": round((drift_count / total) * 100, 2) if total else 0,
    }


@app.get("/logs/recent")
def get_recent_logs(limit: int = 10):
    log_file = LOG_DIR / "predictions.jsonl"

    if not log_file.exists():
        return {"logs": []}

    logs = []
    with open(log_file) as f:
        lines = f.readlines()[-limit:]

    for line in lines:
        try:
            logs.append(json.loads(line))
        except:
            continue

    return {"logs": logs}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )