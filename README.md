
# 🐦 Tweet Analytics - Virality Prediction API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready FastAPI application that predicts tweet virality using a hybrid ensemble model combining gradient boosting, random forests, and NMF-based text analysis.

## 🎯 Overview

Tweet Analytics leverages advanced machine learning to predict how many retweets and likes a tweet will receive. The system combines three complementary prediction models:

- **🚀 Gradient Boosting (XGBoost)** - Embedding-based predictions
- **🌲 Random Forest** - Embedding-based predictions  
- **📊 NMF + Linear Regression** - Text feature-based predictions

The ensemble approach averages predictions from all three models for **robust and reliable virality scoring**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🚀 **FastAPI** | Modern web framework with automatic OpenAPI/Swagger documentation |
| 🤖 **Hybrid Ensemble** | Combines embedding-based and text-based ML models |
| 📊 **Dual Predictions** | Predicts both retweet counts and likes simultaneously |
| 📈 **Monitoring** | Built-in logging and embedding drift detection |
| 🔍 **Drift Detection** | Monitors for out-of-distribution predictions |
| 📝 **Audit Logging** | JSONL format predictions log for compliance |
| ⚡ **Production Ready** | Includes Docker, Gunicorn, and Nginx configs |
| 🎨 **Interactive Dashboard** | Streamlit web interface for real-time predictions |

---

## 📋 Prerequisites

- **Python 3.11+**
- **pip** or **conda** package manager
- **2-4 GB RAM** (for model loading)
- **Optional: CUDA 11.8+** (for GPU acceleration)

---

## 🛠️ Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/kunalpop/tweet-analytics.git
cd tweet-analytics

# Install dependencies
pip install -r requirements.txt
```
---

## 📁 Project Structure

```
tweet-analytics/
│
├── 📄 main.py
│   └── FastAPI application with all endpoints, model loading, and inference
│
├── 🤖 models/
│   ├── gb_pipeline_model.pkl          (125 KB) - XGBoost gradient boosting
│   ├── rf_pipeline_model.pkl          (168 KB) - Random Forest model
│   ├── retweet_pipeline.pkl           (92 KB)  - NMF-based retweet predictor
│   ├── likes_pipeline.pkl             (92 KB)  - NMF-based likes predictor
│   ├── embedder.pkl                   (70 MB)  - Sentence Transformer embedder
│   ├── feature_stats.json             - Embedding statistics for drift detection
│   └── model_metadata.json            - Model versioning and metadata
│
├── 📦 twitnalytics/                   Core package modules
│   ├── __init__.py                    - Package initialization
│   ├── clean.py                       - Text preprocessing utilities
│   ├── io.py                          - Data I/O operations
│   ├── topic_model.py                 - Topic modeling (HDBSCAN + UMAP)
│   └── virality.py                    - Ensemble prediction logic
│
├── 📚 notebooks/                      Jupyter notebooks for analysis
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_tweeter_profiling.ipynb
│   ├── 04_topic_modelling.ipynb
│   ├── 04_topic_modelling_cluster_labellling.ipynb
│   ├── 05_topic_modelling_tabularnn.ipynb
│   ├── 06_Ensemble_Model.ipynb
│   ├── 07_nmf_topic_engagement_analysis.ipynb
│   └── tweets_quality_report.html
│
├── 🔧 scripts/
│   ├── db_utils.py                    - Database utilities
│   └── streamlit_app.py               - Interactive web dashboard
│
├── 📊 data/
│   ├── raw/                           - Original unprocessed data
│   └── processed/                     - Cleaned and transformed data
│
├── 🗄️ db/                             - Database files and schemas
│
├── 📝 logs/
│   ├── inference.log                  - API inference logs
│   ├── predictions.jsonl              - Prediction audit trail
│   └── error.log                      - Error logs
│
├── 📋 Configuration
│   ├── requirements.txt               - pip dependencies
│   ├── environment.yml                - conda environment
│   ├── Procfile                       - Heroku deployment
│   └── .gitignore                     - Git ignore patterns
│
└── __pycache__/                       - Python cache files
```

---

## 🚀 Quick Start

### 1. Start Local Development Server

```bash
# Start with hot reload
uvicorn main:app --reload
```

Server runs at: **`http://localhost:8000`**

### 2. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Run Interactive Dashboard

```bash
streamlit run scripts/streamlit_app.py
```

---

## 📡 API Endpoints

### Health & Status

#### `GET /`
Root endpoint - Service health check
```bash
curl http://localhost:8000/
```
Response:
```json
{
  "status": "healthy",
  "message": "Virality Prediction API is running",
  "models": ["XGBoost", "Random Forest", "NMF + Linear"]
}
```

#### `GET /health`
Detailed health status
```bash
curl http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "total_predictions": 42
}
```

#### `GET /model-info`
Model architecture and configuration
```bash
curl http://localhost:8000/model-info
```

### Predictions

#### `POST /predict`
Main prediction endpoint - **Predict virality for a tweet**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Just launched my new ML project! #MachineLearning"}'
```

**Request:**
```json
{
  "text": "Just launched my new ML project! #MachineLearning"
}
```

**Response:**
```json
{
  "prediction_id": "twt_20260405_120000_000001",
  "timestamp": "2026-04-05T12:00:00.000000",
  "xgboost_prediction": 245,
  "random_forest_prediction": 234,
  "nmf_prediction": 256,
  "ensemble_prediction": 245,
  "predicted_likes": 890,
  "drift_warnings": []
}
```

**Input Constraints:**
- Text: 1-280 characters (standard tweet length)
- Required field
- Automatically cleaned (URLs, mentions, special chars removed)

### Analytics & Logs

#### `GET /logs/summary`
Aggregate prediction statistics
```bash
curl http://localhost:8000/logs/summary
```
Response:
```json
{
  "total_predictions": 150,
  "avg_retweets": 234,
  "avg_likes": 567,
  "drift_rate": 2.34
}
```

#### `GET /logs/recent?limit=10`
Recent predictions with pagination
```bash
curl http://localhost:8000/logs/recent?limit=10
```
---

## ☁️ Cloud Deployment

### Deploy On Render.com

```bash
# 2. Go to Render.com and login
# https://render.com

# 3. Click "New +" → Web Service

# 4. Connect your GitHub repository

# 5. Configure service:
# Name: your-app-name (tweet-analytics)
# Environment: Python 3
# Build Command: pip install -r requirements.txt

# Start Command:
uvicorn app:app --host 0.0.0.0 --port 10000
```
---
## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Response Time** | < 100 ms |
| **Requests/Second** | 1000 |
| **Max Concurrent Users** | 500 |
| **Embedding Dimension** | 384 |
| **Memory Usage** | 2-4 GB |

---

## 🔐 Security Best Practices

- ✅ **HTTPS Only** - Always use HTTPS in production
- ✅ **Rate Limiting** - Implement rate limiting (e.g., 100 req/min)
- ✅ **Input Validation** - Tweet text: 1-280 characters
- ✅ **Environment Variables** - Store secrets in `.env`
- ✅ **Audit Logging** - All predictions logged to JSONL
- ✅ **CORS Configuration** - Restrict origins in production
- ✅ **API Authentication** - Consider adding API key validation

### Environment Variables

```bash
# .env file
PORT=8000
LOG_LEVEL=INFO
MODEL_DIR=./models
DATABASE_URL=postgresql://user:pass@localhost/db
API_KEY_SECRET=your-secret-key
```
---

## 📊 Monitoring & Observability

### Log Files

```
logs/
├── inference.log          # All API requests and responses
├── predictions.jsonl      # Prediction audit trail (JSON Lines format)
└── error.log              # Error logs only
```

### Example Prediction Log Entry

```json
{
  "timestamp": "2026-04-05T12:00:00.000000",
  "prediction_id": "twt_20260405_120000_000001",
  "text": "Great machine learning insight!",
  "retweets": 245,
  "likes": 890,
  "drift": []
}
```

### View Logs

```bash
# Real-time logs
tail -f logs/inference.log

# Recent predictions
tail -20 logs/predictions.jsonl | jq .

# Filter by prediction ID
grep "twt_20260405" logs/predictions.jsonl
```

---

## 🚀 Twit-Nalytics Streamlit Dashboard

Twit-Nalytics is an interactive Streamlit-based analytics platform for exploring tweet data, performing topic modeling, and predicting tweet virality using machine learning. Some of the features of the dashboard are as follows. 

- Data Loading – Import tweet data from SQLite databases
- Overview Analytics – Visualize tweet trends and top hashtags
- Topic Modeling – Discover themes using BERTopic
- User Insights – Identify most active users
- Virality Prediction – Predict if a tweet will go viral (API / ML / heuristic)
- Model Training – Train custom virality models on your data

---
## 🏗️ Architecture

                     ┌──────────────┐
                     │     Text     │
                     └──────┬───────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
 ┌───────────────┐                      ┌──────────────┐
 │ Text Embedder │                      │    TF-IDF    │
 └──────┬────────┘                      └──────┬───────┘
        │                                      │
   ┌────┴─────────┐                     ┌───────┴────────────┐
   ▼              ▼                     ▼                    ▼
┌────────────┐ ┌────────────────┐  ┌────────────────────────────┐
│ Random     │ │ Gradient       │  │ Non-negative Matrix        │
│ Forest     │ │ Boosting       │  │ Factorization (NMF)        │
│ Predictor  │ │ Predictor      │  │                            │
└────┬───────┘ └──────┬─────────┘  └──────────────┬─────────────┘
     │                │                           │
     └────────┬───────┴───────────────┬───────────┘
              ▼                       ▼
          ┌────────────────────────────────┐
          │      Ensemble Averaging        │
          └──────────────┬─────────────────┘
                         ▼
                  ┌──────────────┐
                  │  Retweets    │
                  └──────────────┘
---
## 📦 Dependencies

### Core Framework
- **fastapi** (0.111.0) - Web framework
- **uvicorn** (0.30.1) - ASGI server
- **pydantic** (2.7.4) - Data validation

### Machine Learning & NLP
- **scikit-learn** (1.4.2) - ML models (XGBoost, Random Forest)
- **sentence-transformers** (2.7.0) - Text embeddings
- **transformers** (4.41.2) - NLP models
- **torch** (2.2.2) - Deep learning framework
- **numpy** (1.26.4) - Numerical computing
- **scipy** (1.12.0) - Scientific computing

### Data Processing & Analytics
- **pandas** (2.2.2) - Data manipulation
- **joblib** (1.4.2) - Model serialization
- **umap-learn** (0.5.5) - Dimensionality reduction
- **hdbscan** (0.8.33) - Clustering algorithm
---

**Last Updated:** April 5, 2026

**Version:** 2.0.0

---
