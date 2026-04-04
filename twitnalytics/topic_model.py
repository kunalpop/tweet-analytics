from pathlib import Path
from typing import Optional, Tuple, List, Union
from bertopic import BERTopic
from umap import UMAP
import hdbscan
import pandas as pd
import numpy as np
import os
from typing import Iterable


def build_topic_model(
    embedding_model: str = "all-MiniLM-L6-v2",
    n_neighbors: int = 15,
    n_components: int = 5,
    min_cluster_size: int = 10,
    min_samples: int = 1,
    cluster_selection_method: str = "eom",
    random_state: int = 42,
) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
    )
    return topic_model


def fit_topics(
    texts: List[str],
    model: Optional[BERTopic] = None,
) -> Tuple[BERTopic, np.ndarray, np.ndarray]:
    if model is None:
        model = build_topic_model()
    topics, probs = model.fit_transform(texts)
    return model, np.array(topics), np.array(probs) if probs is not None else np.empty((len(texts), 0))


def _generate_html_report(
    model: BERTopic,
    texts: list,
    topics: np.ndarray,
    outdir: Path,
) -> None:
    info = model.get_topic_info()
    n_docs = len(texts)
    outliers = int((topics == -1).sum()) if topics is not None and topics.size > 0 else 0
    topic_ids: Iterable[int] = [int(t) for t in info["Topic"].tolist() if int(t) != -1]
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'><title>Topic Report</title>")
    lines.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;max-width:1100px;margin:auto}h1{margin-top:0}.topic{border:1px solid #ddd;padding:12px;border-radius:8px;margin:12px 0}.words{color:#333;font-size:14px}.examples li{margin:6px 0}.viz{border:1px solid #eee;border-radius:8px;margin:16px 0;padding:8px}</style>")
    lines.append("</head><body>")
    lines.append(f"<h1>BERTopic Report</h1>")
    try:
        um = model.umap_model
        hm = model.hdbscan_model
        param_line = f"UMAP: neighbors={getattr(um, 'n_neighbors', '?')}, components={getattr(um, 'n_components', '?')} | HDBSCAN: min_cluster_size={getattr(hm, 'min_cluster_size', '?')}, min_samples={getattr(hm, 'min_samples', '?')}, selection={getattr(hm, 'cluster_selection_method', '?')}"
    except Exception:
        param_line = ""
    lines.append(f"<p>Documents: {n_docs} | Topics (excl. -1): {len(topic_ids)} | Outliers: {outliers}</p>")
    if param_line:
        lines.append(f"<p>{param_line}</p>")
    links = []
    try:
        fig_docs = model.visualize_documents(texts, hide_annotations=True)
        fig_docs.write_html(str(outdir / "documents_scatter.html"), include_plotlyjs="cdn", full_html=True)
        links.append("documents_scatter.html")
    except Exception:
        pass
    try:
        fig_bar = model.visualize_barchart(top_n_topics=10)
        fig_bar.write_html(str(outdir / "topics_barchart.html"), include_plotlyjs="cdn", full_html=True)
        links.append("topics_barchart.html")
    except Exception:
        pass
    try:
        fig_h = model.visualize_hierarchy()
        fig_h.write_html(str(outdir / "hierarchy.html"), include_plotlyjs="cdn", full_html=True)
        links.append("hierarchy.html")
    except Exception:
        pass
    if links:
        lines.append("<div class='viz'><h2>Interactive Visuals</h2><ul>")
        for l in links:
            lines.append(f"<li><a href='{l}' target='_blank'>{l}</a></li>")
        lines.append("</ul></div>")
    for tid in topic_ids:
        words_weights = model.get_topic(tid) or []
        top_words = [w for w, _ in words_weights[:10]]
        idx = np.where(topics == tid)[0] if topics is not None and topics.size > 0 else np.array([], dtype=int)
        ex_idx = idx[:3]
        examples = [texts[i] for i in ex_idx] if ex_idx.size > 0 else []
        lines.append("<div class='topic'>")
        lines.append(f"<h2>Topic {tid}</h2>")
        lines.append(f"<div class='words'><strong>Top words:</strong> {', '.join(top_words)}</div>")
        if examples:
            lines.append("<div><strong>Examples:</strong><ul class='examples'>")
            for ex in examples:
                esc = ex.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                lines.append(f"<li>{esc}</li>")
            lines.append("</ul></div>")
        lines.append("</div>")
    lines.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    model: BERTopic,
    texts: List[str],
    topics: np.ndarray,
    probs: np.ndarray,
    outdir: Union[str, Path],
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    topic_info = model.get_topic_info()
    topic_info.to_csv(outdir / "topics_summary.csv", index=False)
    if isinstance(probs, np.ndarray) and probs.size > 0:
        if probs.ndim == 2 and probs.shape[1] > 0:
            max_prob = probs.max(axis=1)
        elif probs.ndim == 1:
            max_prob = probs
        else:
            max_prob = np.zeros(len(texts))
    else:
        max_prob = np.zeros(len(texts))
    if max_prob.shape[0] != len(texts):
        if max_prob.shape[0] > len(texts):
            max_prob = max_prob[: len(texts)]
        else:
            pad = np.zeros(len(texts) - max_prob.shape[0])
            max_prob = np.concatenate([max_prob, pad])
    df = pd.DataFrame({"text": texts, "topic": topics, "probability": max_prob})
    df.to_csv(outdir / "documents_topics.csv", index=False)
    model.save(outdir / "bertopic_model")
    _generate_html_report(model, texts, topics, outdir)
