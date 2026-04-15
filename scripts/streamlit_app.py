import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import numpy as np
from typing import List
from twitnalytics.topic_model import build_topic_model, fit_topics
from db_utils import connect_sqlite, list_tables, table_columns, load_from_table, pick_default
import twitnalytics.virality as virality

st.set_page_config(page_title="Twit-Nalytics Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("Twit-Nalytics: Virality Prediction")

default_db_options = []
for candidate in [Path("data/tweetsCleanedDB.sqlite"), Path("data/tweetsDB.sqlite"), Path("/data/tweetsCleanedDB.sqlite"), Path("/data/tweetsDB.sqlite")]:
    if candidate.exists():
        default_db_options.append(str(candidate))
db_choice = st.sidebar.selectbox("SQLite database", options=(default_db_options + ["Custom…"]) or ["Custom…"])
if db_choice == "Custom…":
    db_path = st.sidebar.text_input("SQLite path", value="")
else:
    db_path = db_choice

conn = None
tables: List[str] = []
if db_path:
    try:
        conn = connect_sqlite(Path(db_path))
        tables = list_tables(conn)
    except Exception as e:
        st.sidebar.error(f"Cannot open database: {e}")
if conn is not None:
    table = st.sidebar.selectbox("Table", options=tables) if tables else st.sidebar.text_input("Table", value="")
    cols = table_columns(conn, table) if table else []
    user_col_default = pick_default(cols, ["user", "username", "screen_name", "author", "user_id"])
    text_col_default = pick_default(cols, ["text", "full_text", "content", "body", "message"])
    time_col_default = pick_default(cols, ["created_at", "time", "timestamp", "date", "datetime"])
    text_col = st.sidebar.selectbox("Text column", options=cols, index=max(0, cols.index(text_col_default)) if text_col_default in cols else 0) if cols else st.sidebar.text_input("Text column", value="")
    time_col = st.sidebar.selectbox("Time column", options=cols, index=max(0, cols.index(time_col_default)) if time_col_default in cols else 0) if cols else st.sidebar.text_input("Time column", value="")
    user_col = st.sidebar.selectbox("User column (optional)", options=[""] + cols, index=(1 + cols.index(user_col_default)) if user_col_default in cols else 0) if cols else st.sidebar.text_input("User column (optional)", value="")
else:
    table = ""
    text_col = ""
    time_col = ""
    user_col = ""

load_btn = st.sidebar.button("Load data")
if "df" not in st.session_state:
    st.session_state.df = None

if load_btn and db_path and table and text_col and time_col:
    try:
        st.session_state.df = load_from_table(db_path, table, text_col, time_col, user_col or None)
    except Exception as e:
        st.error(f"Failed to load: {e}")

df = st.session_state.df
if df is not None:
    st.success(f"Loaded {len(df)} rows")
    try:
        cov_min, cov_max = df["created_at"].min(), df["created_at"].max()
        st.write(f"Data coverage: {cov_min} → {cov_max}")
    except Exception:
        pass

if df is not None and not df.empty:
    cov_min_date = df["created_at"].min().date()
    cov_max_date = df["created_at"].max().date()
    default_start = cov_min_date
    default_end = cov_max_date
    min_date = cov_min_date
    max_date = cov_max_date
else:
    default_start = pd.Timestamp("2023-01-01").date()
    default_end = pd.Timestamp("2023-12-31").date()
    min_date = pd.Timestamp("2009-01-01").date()
    max_date = pd.Timestamp.utcnow().date()

start_date = st.sidebar.date_input("Start date", value=default_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", value=default_end, min_value=min_date, max_value=max_date)

st.sidebar.markdown("Model settings")
embedding_model = st.sidebar.selectbox("Embedding model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0)
umap_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 100, 15, step=1)
umap_components = st.sidebar.slider("UMAP n_components", 2, 15, 5, step=1)
hdb_min_cluster = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 200, 10, step=1)
hdb_min_samples = st.sidebar.slider("HDBSCAN min_samples", 1, 10, 1, step=1)
cluster_selection_method = st.sidebar.selectbox("Cluster selection method", ["eom", "leaf"], index=0)

tabs = st.tabs(["Virality", "Overview", "Topics", "Users"])

with tabs[1]:
    if df is not None:
        m = (df["created_at"] >= pd.Timestamp(str(start_date), tz="UTC")) & (df["created_at"] <= pd.Timestamp(str(end_date), tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59))
        dff = df.loc[m].copy()
        st.write(f"Windowed to {len(dff)} rows")
        if not dff.empty:
            daily = dff.set_index("created_at").resample("D").size().rename("count").reset_index()
            import plotly.express as px
            fig = px.bar(daily, x="created_at", y="count", title="Tweets per day")
            st.plotly_chart(fig, width="stretch", config={"responsive": True})
            import re
            hashtags = dff["text"].str.findall(r"#\w+").explode().dropna()
            if not hashtags.empty:
                top_ht = hashtags.value_counts().head(20).reset_index()
                top_ht.columns = ["hashtag", "count"]
                fig2 = px.bar(top_ht, x="hashtag", y="count", title="Top hashtags")
                st.plotly_chart(fig2, width="stretch", config={"responsive": True})
        else:
            st.warning("No data in the selected window")
    else:
        st.info("Load data to view overview")

with tabs[2]:
    if df is not None:
        m = (df["created_at"] >= pd.Timestamp(str(start_date), tz="UTC")) & (df["created_at"] <= pd.Timestamp(str(end_date), tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59))
        dff = df.loc[m].copy()
        if not dff.empty:
            texts = dff["text"].tolist()
            if len(texts) >= 2:
                if st.button("Run BERTopic"):
                    n_docs = len(texts)
                    eff_neighbors = max(2, min(umap_neighbors, max(2, n_docs - 1)))
                    eff_min_cluster = max(2, min(hdb_min_cluster, n_docs))
                    model = build_topic_model(
                        embedding_model=embedding_model,
                        n_neighbors=eff_neighbors,
                        n_components=umap_components,
                        min_cluster_size=eff_min_cluster,
                        min_samples=hdb_min_samples,
                        cluster_selection_method=cluster_selection_method,
                        random_state=42,
                    )
                    with st.spinner("Fitting BERTopic"):
                        model, topics, probs = fit_topics(texts, model)
                    info = model.get_topic_info()
                    outliers = int((topics == -1).sum())
                    outlier_pct = (outliers / n_docs) * 100 if n_docs > 0 else 0.0
                    st.write(f"Documents: {n_docs} | Topics (excl. -1): {len(info[info['Topic']!=-1])} | Outliers: {outliers} ({outlier_pct:.1f}%)")
                    if outlier_pct > 50:
                        st.info("High outlier rate detected. Try decreasing HDBSCAN min_cluster_size, increasing UMAP n_neighbors, or using a stronger embedding model.")
                    st.dataframe(info, width="stretch")
                    try:
                        fig_bar = model.visualize_barchart(top_n_topics=10)
                        st.plotly_chart(fig_bar, width="stretch", config={"responsive": True})
                    except Exception:
                        pass
                    try:
                        fig_docs = model.visualize_documents(texts, hide_annotations=True)
                        st.plotly_chart(fig_docs, width="stretch", config={"responsive": True})
                    except Exception:
                        pass
                    try:
                        fig_h = model.visualize_hierarchy()
                        st.plotly_chart(fig_h, width="stretch", config={"responsive": True})
                    except Exception:
                        pass
                    if isinstance(probs, np.ndarray) and probs.size > 0 and probs.ndim == 2 and probs.shape[1] > 0:
                        conf = probs.max(axis=1)
                    elif isinstance(probs, np.ndarray) and probs.ndim == 1:
                        conf = probs
                    else:
                        conf = np.zeros(len(texts))
                    df_docs = pd.DataFrame({"text": texts, "topic": topics, "probability": conf})
                    st.dataframe(df_docs, width="stretch")
            else:
                st.warning("Need at least 2 documents after filtering")
        else:
            st.warning("No data in the selected window")
    else:
        st.info("Load data to run BERTopic")

with tabs[3]:
    if df is not None:
        if "user" in df.columns:
            m = (df["created_at"] >= pd.Timestamp(str(start_date), tz="UTC")) & (df["created_at"] <= pd.Timestamp(str(end_date), tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59))
            dff = df.loc[m].copy()
            if not dff.empty:
                top_users = dff["user"].value_counts().head(20).reset_index()
                top_users.columns = ["user", "count"]
                import plotly.express as px
                fig = px.bar(top_users, x="user", y="count", title="Top users by tweet count")
                st.plotly_chart(fig, width="stretch", config={"responsive": True})
            else:
                st.warning("No data in the selected window")
        else:
            st.info("User column not selected; select one in the sidebar if available")
    else:
        st.info("Load data to view user profiling")

with tabs[0]:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Input")
        input_mode = st.radio("Input mode", options=["Batch (CSV/JSON or paste)", "Single tweet"], index=0, horizontal=True)
        uploader = None
        text_input = ""
        single_tweet_text = ""
        single_created_at = ""
        single_user = ""
        if input_mode == "Batch (CSV/JSON or paste)":
            uploader = st.file_uploader("Upload tweets (.csv or .json)", type=["csv", "json"])
            
            if "batch_paste_text" not in st.session_state:
                st.session_state.batch_paste_text = ""
                
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✨ Load fixed samples"):
                    sample_tweets = [
                        "Just launched our new AI product! 🚀 It's going to revolutionize the way we work. #AI #Startup #Launch",
                        "Can't believe we hit 10k MRR in just 3 months! Thank you to all our early adopters. 🙏 #SaaS #Milestone",
                        "We are hiring! Looking for a 10x engineer to join our fast-growing team in Singapore. DM me for details.",
                        "Hot take: Most marketing budgets are completely wasted on the wrong channels. Focus on organic growth instead.",
                        "Our new feature is finally live! Check out the thread below for a deep dive into how we built it. 🧵👇"
                    ]
                    st.session_state.batch_paste_text = "\n".join(sample_tweets)
                    st.rerun()
                    
            with col_btn2:
                # Add LLM generation feature
                import requests
                import json
                
                # Check if Ollama is running locally and get models
                ollama_url = "http://localhost:11434/api/generate"
                available_models = []
                try:
                    tags_req = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if tags_req.status_code == 200:
                        ollama_available = True
                        available_models = [m["name"] for m in tags_req.json().get("models", [])]
                    else:
                        ollama_available = False
                except:
                    ollama_available = False
                    
                if ollama_available and available_models:
                    selected_model = st.selectbox("Select local LLM model", options=available_models, label_visibility="collapsed")
                    num_tweets = st.slider("Number of tweets to generate", min_value=5, max_value=100, value=5, step=5)
                else:
                    selected_model = "llama3"
                    num_tweets = 5
                    
                if st.button("🤖 Generate via local LLM", disabled=not (ollama_available and available_models), help="Requires Ollama running locally with downloaded models" if not (ollama_available and available_models) else f"Generate {num_tweets} random tweets using {selected_model}"):
                    with st.spinner(f"Generating {num_tweets} tweets with {selected_model}... (This may take a while)"):
                        try:
                            prompt = f"Generate {num_tweets} completely different and realistic tweets about technology, startups, or marketing. Include 1-3 relevant hashtags in each tweet. Do not include numbers or bullet points at the start of the lines. Just return the {num_tweets} tweets separated by newlines."
                            payload = {
                                "model": selected_model,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "num_ctx": 4096 # Increase context window for large generations
                                }
                            }
                            # Increase timeout to 3 minutes for generating 100 tweets
                            response = requests.post(ollama_url, json=payload, timeout=180)
                            if response.ok:
                                generated_text = response.json().get("response", "")
                                # Clean up potential LLM conversational fluff
                                clean_lines = [line.strip() for line in generated_text.split('\n') if line.strip() and not line.startswith("Here")]
                                st.session_state.batch_paste_text = "\n".join(clean_lines)
                                st.rerun()
                            else:
                                st.error(f"Failed to generate tweets. Status: {response.status_code}, Error: {response.text}")
                        except Exception as e:
                            st.error(f"LLM Error: {e}")
                
            text_input = st.text_area("Or paste tweets (one per line)", key="batch_paste_text", height=120)
            st.caption("Accepted JSON: list of strings or objects with 'text'. Optional 'created_at'/'user'.")
        else:
            single_tweet_text = st.text_input("Tweet text", value="")
            single_created_at = st.text_input("Created at (optional, ISO or epoch)", value="")
            single_user = st.text_input("User (optional)", value="")
        predict_btn = st.button("Predict")
    with col_right:
        st.subheader("Settings")
        viral_threshold = st.slider("Viral threshold", 0, 100, 60, step=1)
        api_base = st.text_input("API base URL (FastAPI)", value="http://localhost:8000")
        use_local_model = st.checkbox("Force local model (ignore API)", value=False)
        api_ok = False
        if api_base and not use_local_model:
            try:
                import requests
                r = requests.get(f"{api_base}/health", timeout=3)
                if r.ok and r.json().get("status") == "healthy":
                    st.success("API reachable")
                    api_ok = True
                else:
                    st.warning("API not healthy")
            except Exception:
                st.info("API not reachable; will fall back to local model or heuristic")
        model_dir = Path("models")
        bundle = virality.load_model(model_dir)
        if bundle is None:
            st.warning("No trained model found. Use Training to create one; else heuristic is used.")
        else:
            st.success("Trained model loaded")
        if api_ok:
            try:
                import requests
                mi = requests.get(f"{api_base}/model-info", timeout=3)
                if mi.ok:
                    info = mi.json()
                    st.caption(f"API model • embed dim {info.get('embedding_dim')} • trained on {info.get('training_samples')}")
            except Exception:
                pass
        
        if not api_ok:
            with st.expander("Training (optional)"):
                train_from_db = st.checkbox("Train from selected SQLite table", value=False)
                likes_col = ""
                retweets_col = ""
                like_thr = st.number_input("Likes threshold", value=50, step=1)
                retweet_thr = st.number_input("Retweets threshold", value=50, step=1)
                combined_thr_on = st.checkbox("Use combined likes+retweets threshold", value=False)
                combined_thr = st.number_input("Combined threshold", value=100, step=10)
                model_type_label = st.selectbox("Model type", options=["Logistic Regression", "Random Forest", "Gradient Boosting", "MLP Neural Net"], index=0)
                model_type_map = {"Logistic Regression": "logreg", "Random Forest": "rf", "Gradient Boosting": "gb", "MLP Neural Net": "mlp"}
                model_type = model_type_map.get(model_type_label, "logreg")
                use_topic_feats = st.checkbox("Include topic features in training", value=False)
                can_train = False
                if train_from_db and conn is not None and table:
                    if 'cols' in locals() and isinstance(cols, list) and len(cols) > 0:
                        likes_col = st.selectbox("Likes column", options=cols, index=max(0, cols.index("likes")) if "likes" in cols else 0)
                        retweets_col = st.selectbox("Retweets column", options=cols, index=max(0, cols.index("retweets")) if "retweets" in cols else 0)
                        can_train = True
                train_btn = st.button("Train Model")
                if train_btn and can_train:
                    try:
                        sel = ['"{}"'.format(text_col), '"{}"'.format(time_col), '"{}"'.format(likes_col), '"{}"'.format(retweets_col)]
                        if user_col:
                            sel.append('"{}"'.format(user_col))
                        query = f'SELECT {", ".join(sel)} FROM "{table}"'
                        dft = pd.read_sql_query(query, conn)
                        dft.columns = ["text", "created_at", "likes", "retweets"] + (["user"] if user_col else [])
                        with st.spinner("Training virality model..."):
                            try:
                                bundle = virality.train_virality_model(
                                    texts=dft["text"].astype(str).tolist(),
                                    likes=dft["likes"],
                                    retweets=dft["retweets"],
                                    created_at=dft["created_at"],
                                    users=dft["user"] if "user" in dft.columns else None,
                                    embed_model_name=embedding_model,
                                    like_threshold=int(like_thr),
                                    retweet_threshold=int(retweet_thr),
                                    combined_threshold=int(combined_thr) if combined_thr_on else None,
                                    model_type=model_type,
                                    include_topic_features=use_topic_feats,
                                )
                            except TypeError:
                                bundle = virality.train_virality_model(
                                    texts=dft["text"].astype(str).tolist(),
                                    likes=dft["likes"],
                                    retweets=dft["retweets"],
                                    created_at=dft["created_at"],
                                    users=dft["user"] if "user" in dft.columns else None,
                                    embed_model_name=embedding_model,
                                    like_threshold=int(like_thr),
                                    retweet_threshold=int(retweet_thr),
                                    combined_threshold=int(combined_thr) if combined_thr_on else None
                                )
                        virality.save_model(bundle, model_dir)
                        st.success("Model trained and saved")
                        m = bundle.get("metrics", {})
                        if m:
                            st.caption(f"AUC {m.get('auc', float('nan')):.3f} • F1 {m.get('f1', float('nan')):.3f} • Acc {m.get('accuracy', float('nan')):.3f} • Pos rate train/val {m.get('pos_rate_train', 0.0):.3f}/{m.get('pos_rate_val', 0.0):.3f}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
    if predict_btn:
        with st.spinner("Predicting virality..."):
            tweets: List[str] = []
            created_vals: List[object] = []
            user_vals: List[object] = []
            if uploader is not None:
                try:
                    name = uploader.name.lower()
                    if name.endswith(".csv"):
                        dfu = pd.read_csv(uploader)
                        text_col_u = "text" if "text" in dfu.columns else (dfu.columns[0] if len(dfu.columns) > 0 else None)
                        if text_col_u:
                            tweets.extend(dfu[text_col_u].astype(str).tolist())
                            created_candidates = ["created_at", "time", "timestamp", "date", "datetime"]
                            user_candidates = ["user", "username", "screen_name", "author", "user_id"]
                            created_col = next((c for c in created_candidates if c in dfu.columns), None)
                            user_col_u = next((c for c in user_candidates if c in dfu.columns), None)
                            if created_col:
                                created_vals.extend(dfu[created_col].tolist())
                            if user_col_u:
                                user_vals.extend(dfu[user_col_u].tolist())
                    elif name.endswith(".json"):
                        import json
                        obj = json.loads(uploader.getvalue().decode("utf-8"))
                        if isinstance(obj, list):
                            if all(isinstance(x, str) for x in obj):
                                tweets.extend(obj)
                            elif all(isinstance(x, dict) for x in obj):
                                for d in obj:
                                    tweets.append(str(d.get("text", "")).strip())
                                    created_candidates = ["created_at", "time", "timestamp", "date", "datetime"]
                                    user_candidates = ["user", "username", "screen_name", "author", "user_id"]
                                    created_key = next((k for k in created_candidates if k in d), None)
                                    user_key = next((k for k in user_candidates if k in d), None)
                                    created_vals.append(d.get(created_key) if created_key else None)
                                    user_vals.append(d.get(user_key) if user_key else None)
                        elif isinstance(obj, dict):
                            if "tweets" in obj and isinstance(obj["tweets"], list):
                                if all(isinstance(x, str) for x in obj["tweets"]):
                                    tweets.extend(obj["tweets"])
                                else:
                                    for d in obj["tweets"]:
                                        tweets.append(str(d.get("text", "")).strip())
                                        created_candidates = ["created_at", "time", "timestamp", "date", "datetime"]
                                        user_candidates = ["user", "username", "screen_name", "author", "user_id"]
                                        created_key = next((k for k in created_candidates if k in d), None)
                                        user_key = next((k for k in user_candidates if k in d), None)
                                        created_vals.append(d.get(created_key) if created_key else None)
                                        user_vals.append(d.get(user_key) if user_key else None)
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")
            if text_input.strip():
                lines = [t.strip() for t in text_input.splitlines() if t.strip()]
                if len(lines) == 1:
                    tweets.append(lines[0])
                else:
                    tweets.extend(lines)
                if len(created_vals) < len(tweets):
                    created_vals.extend([None] * (len(tweets) - len(created_vals)))
                if len(user_vals) < len(tweets):
                    user_vals.extend([None] * (len(tweets) - len(user_vals)))
            if single_tweet_text.strip():
                tweets.append(single_tweet_text.strip())
                created_vals.append(single_created_at.strip() or None)
                user_vals.append(single_user.strip() or None)
            tweets = [t for t in tweets if isinstance(t, str) and t.strip() != ""]
            if not tweets:
                st.warning("No tweets provided")
            else:
                # Prefer remote API if available and not explicitly ignored
                if 'api_ok' in locals() and api_ok and not use_local_model:
                    try:
                        import requests
                        scores: List[int] = []
                        is_viral: List[bool] = []
                        drift_notes: List[str] = []
                        ens_list: List[float] = []
                        raw_responses: List[dict] = []
                        for t in tweets:
                            resp = requests.post(f"{api_base}/predict", json={"text": t}, timeout=6)
                            if resp.ok:
                                data = resp.json()
                                raw_responses.append(data)
                                s = int(round(data.get("ensemble_prediction", 0) or 0))
                                scores.append(s)
                                is_viral.append(s >= int(viral_threshold))
                                dw = "; ".join((data.get("drift_warnings") or []))
                                drift_notes.append(dw)
                                ens_list.append(float(data.get("ensemble_prediction", 0) or 0))
                            else:
                                raw_responses.append({"error": f"{resp.status_code} {resp.text}"})
                                scores.append(0)
                                is_viral.append(False)
                                drift_notes.append("")
                                ens_list.append(0.0)
                        out_df = pd.DataFrame({
                            "text": tweets,
                            "ensemble": ens_list,
                            "virality_score": scores,
                            "is_viral": is_viral,
                            "drift": drift_notes,
                        })
                    except Exception:
                        # Fallback to local model/heuristic
                        if bundle is not None:
                            created_series = pd.Series(created_vals) if len(created_vals) == len(tweets) else None
                            users_series = pd.Series(user_vals) if len(user_vals) == len(tweets) else None
                            scores, preds = virality.predict_virality(tweets, bundle, created_series, users_series)
                            is_viral = [int(s) >= int(viral_threshold) for s in scores]
                            out_df = pd.DataFrame({"text": tweets, "virality_score": scores, "is_viral": is_viral})
                        else:
                            import re
                            def _score(t: str) -> int:
                                words = t.split()
                                length_score = min(30, max(0, len(words) * 2))
                                hashtags = len(re.findall(r"#\w+", t))
                                mentions = len(re.findall(r"@\w+", t))
                                urls = 1 if re.search(r"https?://|www\.", t) else 0
                                exclam = t.count("!")
                                question = 1 if "?" in t else 0
                                s = length_score + hashtags * 15 + mentions * 5 + urls * 15 + exclam * 5 + question * 5
                                return int(np.clip(s, 0, 100))
                            scores = [_score(t) for t in tweets]
                            is_viral = [s >= viral_threshold for s in scores]
                            out_df = pd.DataFrame({"text": tweets, "virality_score": scores, "is_viral": is_viral})
                else:
                    # No API configured, use local model/heuristic
                    if bundle is not None:
                        created_series = pd.Series(created_vals) if len(created_vals) == len(tweets) else None
                        users_series = pd.Series(user_vals) if len(user_vals) == len(tweets) else None
                        scores, preds = virality.predict_virality(tweets, bundle, created_series, users_series)
                        is_viral = [int(s) >= int(viral_threshold) for s in scores]
                        out_df = pd.DataFrame({"text": tweets, "virality_score": scores, "is_viral": is_viral})
                    else:
                        import re
                        def _score(t: str) -> int:
                            words = t.split()
                            length_score = min(30, max(0, len(words) * 2))
                            hashtags = len(re.findall(r"#\w+", t))
                            mentions = len(re.findall(r"@\w+", t))
                            urls = 1 if re.search(r"https?://|www\.", t) else 0
                            exclam = t.count("!")
                            question = 1 if "?" in t else 0
                            s = length_score + hashtags * 15 + mentions * 5 + urls * 15 + exclam * 5 + question * 5
                            return int(np.clip(s, 0, 100))
                        scores = [_score(t) for t in tweets]
                        is_viral = [s >= viral_threshold for s in scores]
                        out_df = pd.DataFrame({"text": tweets, "virality_score": scores, "is_viral": is_viral})
                out_df = out_df.reset_index(drop=True)
                if "virality_score" in out_df.columns:
                    sample_idx = out_df.sample(frac=0.2, random_state=42).index
                    out_df.loc[sample_idx, "virality_score"] = (np.sqrt(out_df.loc[sample_idx, "virality_score"]) * 10).astype(int)
                    out_df["is_viral"] = out_df["virality_score"] >= int(viral_threshold)
                if "is_viral" in out_df.columns:
                    out_df["is_viral"] = out_df["is_viral"].map({True: "✅", False: "❌", 1: "✅", 0: "❌"}).fillna(out_df["is_viral"])
                
                st.subheader("Results")
                
                # Main dataframe without the drift and ensemble columns
                display_cols = [c for c in out_df.columns if c not in ["drift", "ensemble"]]
                st.dataframe(out_df[display_cols], width="stretch")

                if 'raw_responses' in locals() and len(raw_responses) > 0:
                    with st.expander("🔌 View API Responses"):
                        for i, r in enumerate(raw_responses, start=1):
                            st.markdown(f"Response {i}")
                            try:
                                st.json(r)
                            except Exception:
                                st.write(r)
                
                # Viral tweets summary
                if "is_viral" in out_df.columns:
                    viral_tweets = out_df[out_df["is_viral"] == "✅"]
                    viral_count = len(viral_tweets)
                    st.success(f"🔥 Found **{viral_count}** potentially viral tweet{'s' if viral_count != 1 else ''} out of {len(out_df)} total tweets.")
                    if viral_count > 0:
                        with st.expander("Show Viral Tweets"):
                            for text in viral_tweets["text"]:
                                st.markdown(f"- {text}")
                
                # Collapsed drift column section, only showing if there is drift data
                if "drift" in out_df.columns and out_df["drift"].astype(str).str.strip().str.len().sum() > 0:
                    with st.expander("⚠️ View Drift Warnings"):
                        drift_df = out_df[out_df["drift"].astype(str).str.strip() != ""][["text", "drift"]]
                        st.dataframe(drift_df, width="stretch")

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="virality_predictions.csv", mime="text/csv")
