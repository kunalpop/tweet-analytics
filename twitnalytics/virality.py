from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import joblib
import json
import re

from twitnalytics.topic_model import build_topic_model, fit_topics
from bertopic import BERTopic

def _metadata_features(texts: List[str], created_at: Optional[pd.Series] = None, users: Optional[pd.Series] = None) -> np.ndarray:
    n = len(texts)
    a1 = np.array([len(t.split()) for t in texts], dtype=np.float32).reshape(n, 1)
    a2 = np.array([len(re.findall(r"#\w+", t)) for t in texts], dtype=np.float32).reshape(n, 1)
    a3 = np.array([len(re.findall(r"@\w+", t)) for t in texts], dtype=np.float32).reshape(n, 1)
    a4 = np.array([1 if re.search(r"https?://|www\.", t) else 0 for t in texts], dtype=np.float32).reshape(n, 1)
    a5 = np.array([t.count("!") for t in texts], dtype=np.float32).reshape(n, 1)
    a6 = np.array([1 if "?" in t else 0 for t in texts], dtype=np.float32).reshape(n, 1)
    cols = [a1, a2, a3, a4, a5, a6]
    try:
        if isinstance(created_at, pd.Series):
            hours = pd.to_datetime(created_at, errors="coerce", utc=True).dt.hour.fillna(0).astype(int).to_numpy()
        else:
            hours = np.zeros(n, dtype=np.float32)
    except Exception:
        hours = np.zeros(n, dtype=np.float32)
    cols.append(hours.reshape(n, 1).astype(np.float32))
    try:
        if isinstance(users, pd.Series):
            freq = users.astype(str).value_counts()
            user_freq = users.astype(str).map(freq).fillna(0).to_numpy()
        else:
            user_freq = np.zeros(n, dtype=np.float32)
    except Exception:
        user_freq = np.zeros(n, dtype=np.float32)
    cols.append(user_freq.reshape(n, 1).astype(np.float32))
    return np.concatenate(cols, axis=1)

def _embed(texts: List[str], model_name: str) -> np.ndarray:
    st_model = SentenceTransformer(model_name)
    emb = st_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)

def build_features(texts: List[str], created_at: Optional[pd.Series], users: Optional[pd.Series], embed_model_name: str) -> np.ndarray:
    meta = _metadata_features(texts, created_at, users)
    emb = _embed(texts, embed_model_name)
    return np.concatenate([emb, meta], axis=1)

def _topic_features_with_model(texts: List[str], topic_model: BERTopic) -> np.ndarray:
    topics, probs = topic_model.transform(texts)
    n = len(texts)
    if isinstance(probs, np.ndarray) and probs.size > 0:
        if probs.ndim == 2 and probs.shape[1] > 0:
            top_prob = probs.max(axis=1)
        elif probs.ndim == 1:
            top_prob = probs
        else:
            top_prob = np.zeros(n, dtype=np.float32)
    else:
        top_prob = np.zeros(n, dtype=np.float32)
    outlier = (np.array(topics) == -1).astype(np.float32)
    return np.stack([top_prob.astype(np.float32), outlier], axis=1)

def _fit_topic_model_and_features(texts: List[str], embed_model_name: str) -> Tuple[BERTopic, np.ndarray]:
    model = build_topic_model(embedding_model=embed_model_name)
    model, topics, probs = fit_topics(texts, model)
    n = len(texts)
    if isinstance(probs, np.ndarray) and probs.size > 0:
        if probs.ndim == 2 and probs.shape[1] > 0:
            top_prob = probs.max(axis=1)
        elif probs.ndim == 1:
            top_prob = probs
        else:
            top_prob = np.zeros(n, dtype=np.float32)
    else:
        top_prob = np.zeros(n, dtype=np.float32)
    outlier = (np.array(topics) == -1).astype(np.float32)
    feats = np.stack([top_prob.astype(np.float32), outlier], axis=1)
    return model, feats

def _make_estimator(model_type: str, random_state: int):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_type == "gb":
        return GradientBoostingClassifier(random_state=random_state)
    if model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, random_state=random_state)
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

def train_virality_model(
    texts: List[str],
    likes: pd.Series,
    retweets: pd.Series,
    created_at: Optional[pd.Series],
    users: Optional[pd.Series],
    embed_model_name: str = "all-mpnet-base-v2",
    like_threshold: int = 50,
    retweet_threshold: int = 50,
    combined_threshold: Optional[int] = None,
    model_type: str = "logreg",
    include_topic_features: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    if combined_threshold is not None:
        y = ((likes.fillna(0).astype(float) + retweets.fillna(0).astype(float)) >= combined_threshold).astype(int).to_numpy()
    else:
        y = ((likes.fillna(0).astype(float) >= like_threshold) | (retweets.fillna(0).astype(float) >= retweet_threshold)).astype(int).to_numpy()
    X = build_features(texts, created_at, users, embed_model_name)
    topic_model = None
    if include_topic_features:
        topic_model, tfeats = _fit_topic_model_and_features(texts, embed_model_name)
        X = np.concatenate([X, tfeats], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    est = _make_estimator(model_type, random_state)
    est.fit(X_train, y_train)
    y_val_prob = est.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_val, y_val_prob)) if len(np.unique(y_val)) > 1 else float("nan"),
        "f1": float(f1_score(y_val, y_val_pred)) if len(np.unique(y_val)) > 1 else float("nan"),
        "precision": float(precision_score(y_val, y_val_pred)) if len(np.unique(y_val)) > 1 else float("nan"),
        "recall": float(recall_score(y_val, y_val_pred)) if len(np.unique(y_val)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_val, y_val_pred)),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_val": float(y_val.mean()),
    }
    bundle = {"estimator": est, "embed_model_name": embed_model_name, "model_type": model_type, "metrics": metrics, "use_topic_features": include_topic_features}
    if include_topic_features and topic_model is not None:
        bundle["topic_model"] = topic_model
    return bundle

def save_model(bundle: Dict[str, Any], outdir: Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, outdir / "virality_model.joblib")
    cfg = {"embed_model_name": bundle.get("embed_model_name", ""), "model_type": bundle.get("model_type", ""), "use_topic_features": bundle.get("use_topic_features", False)}
    if "topic_model" in bundle:
        tfile = outdir / "virality_topics_model"
        tm: BERTopic = bundle["topic_model"]
        tm.save(str(tfile))
        cfg["topic_model_dir"] = "virality_topics_model"
    (outdir / "virality_model.json").write_text(json.dumps(cfg))

def load_model(path: Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "virality_model.joblib"
    if not p.exists():
        return None
    bundle = joblib.load(p)
    return bundle

def predict_virality(
    texts: List[str],
    bundle: Dict[str, Any],
    created_at: Optional[pd.Series] = None,
    users: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    embed_model_name = bundle.get("embed_model_name", "all-mpnet-base-v2")
    X = build_features(texts, created_at, users, embed_model_name)
    if bundle.get("use_topic_features", False):
        # Try to load BERTopic model from disk based on config
        topic_model_file = (Path("models") / "virality_topics_model")
        try:
            if topic_model_file.exists():
                from bertopic import BERTopic
                tm = BERTopic.load(str(topic_model_file))
                tfeats = _topic_features_with_model(texts, tm)
            else:
                tfeats = np.zeros((len(texts), 2), dtype=np.float32)
        except Exception:
            tfeats = np.zeros((len(texts), 2), dtype=np.float32)
        X = np.concatenate([X, tfeats], axis=1)
    est = bundle["estimator"]
    prob = est.predict_proba(X)[:, 1]
    scores = (prob * 100.0).astype(int)
    preds = (scores >= 50).astype(bool)
    return scores, preds
