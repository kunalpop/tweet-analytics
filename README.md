
# Tweet Analytics - Virality Prediction API

A production-ready FastAPI application that predicts tweet virality using a hybrid ensemble model combining gradient boosting, random forests, and NMF-based text analysis.

## 🎯 Overview

Tweet Analytics predicts how many retweets and likes a tweet will receive using advanced machine learning techniques. The system combines three complementary prediction models:

- **Gradient Boosting (XGBoost)** - Embedding-based predictions
- **Random Forest** - Embedding-based predictions  
- **NMF + Linear Regression** - Text feature-based predictions

The ensemble approach averages predictions from all three models for robust virality scoring.

## ✨ Features

- 🚀 **FastAPI** - Modern, fast web framework with automatic OpenAPI documentation
- 🤖 **Hybrid Ensemble** - Combines embedding-based and text-based models
- 📊 **Dual Predictions** - Predicts both retweet counts and likes
- 📈 **Monitoring & Observability** - Built-in logging and drift detection
- 🔍 **Embedding Drift Detection** - Monitors for out-of-distribution predictions
- 📝 **Request Logging** - JSONL format predictions log for audit trails
- ⚡ **Production Ready** - Includes deployment configurations

## 📋 Prerequisites

- Python 3.11+
- pip or conda package manager
- FastAPI and dependencies (see requirements.txt)

## 🛠️ Installation

### Option 1: Using pip

```bash
pip install -r requirements.txt

## 📁 Project Structure
tweet-analytics/
├── main.py                      # FastAPI application with prediction endpoints
├── requirements.txt             # Python package dependencies
├── environment.yml              # Conda environment specification
├── Procfile                     # Heroku deployment configuration
├── models/                      # Trained ML models (pkl files)
│   ├── gb_pipeline_model.pkl
│   ├── rf_pipeline_model.pkl
│   ├── retweet_pipeline.pkl
│   ├── likes_pipeline.pkl
│   ├── embedder.pkl
│   └── feature_stats.json
├── notebooks/                   # Jupyter notebooks for exploration
├── data/                        # Training/reference data
├── scripts/                     # Utility scripts
├── db/                          # Database files
├── logs/                        # Application logs
│   ├── inference.log
│   └── predictions.jsonl
└── twitnalytics/               # Main package code
