# IS5126-Twit-Nalytics: BERTopic for Higgs Twitter (July 1–7, 2012)

End-to-end pipeline to load tweets, filter an event window (2012-07-01 to 2012-07-07 UTC), and discover topics automatically by combining Sentence-Transformers embeddings with UMAP (dimensionality reduction) and HDBSCAN (clustering) via BERTopic.

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

if you want to use conda, create a new environment:

```conda
conda create --twit_analytics python=3.11 -y
conda activate twit_analytics
conda install --file requirements.txt -c conda-forge
```

2. Run with the included sample CSV:

```bash
python scripts/run_bertopic.py \
  --input examples/sample_higgs_tweets.csv \
  --text-column text \
  --time-column created_at \
  --start 2012-07-01 \
  --end 2012-07-07 \
  --output outputs/sample-2012-07-01-07
```

3. Inspect outputs:
- `outputs/.../topics_summary.csv` – topic ids, sizes, and representative words
- `outputs/.../documents_topics.csv` – per-document topic and assignment probability
- `outputs/.../bertopic_model` – saved BERTopic model

## Using Your Own Higgs Twitter Data

Supply a CSV path via `--input`. The loader will try to auto-detect columns:
- Text candidates: `text`, `full_text`, `content`, `body`, `message`
- Time candidates: `created_at`, `time`, `timestamp`, `date`, `datetime`

Timestamps can be ISO strings or numeric epochs (seconds or milliseconds). Everything is converted to UTC.

Example:

```bash
python scripts/run_bertopic.py \
  --input data/higgs_tweets.csv \
  --text-column content \
  --time-column created_at \
  --start 2012-07-01 \
  --end 2012-07-07 \
  --output outputs/higgs-2012-07-01-07
```

If your dataset lacks tweet text, consider clustering hashtags or other fields; open an issue or extend the loader.

## Configuration

Key flags:
- `--embedding-model` (default: `all-MiniLM-L6-v2`) – Sentence-Transformers model
- `--n-neighbors` (default: `15`) – UMAP neighbors
- `--n-components` (default: `5`) – UMAP dimensions
- `--min-cluster-size` (default: `10`) – HDBSCAN granularity
- `--random-state` (default: `42`) – reproducibility

Tune `min-cluster-size` for more/fewer topics; adjust UMAP neighbors/components for separation.

## Repository Structure

- [scripts/run_bertopic.py](scripts/run_bertopic.py) – CLI entry point
- [twitnalytics/io.py](twitnalytics/io.py) – robust CSV loader (text, timestamp)
- [twitnalytics/clean.py](twitnalytics/clean.py) – event window filter (UTC)
- [twitnalytics/topic_model.py](twitnalytics/topic_model.py) – BERTopic pipeline and persistence
- [examples/sample_higgs_tweets.csv](examples/sample_higgs_tweets.csv) – small sample for verification

## Troubleshooting

- `ModuleNotFoundError: No module named 'twitnalytics'`  
  The runner adds the repo root to `sys.path`. If you invoke differently, run with `PYTHONPATH=.` or install the package in editable mode:
  ```bash
  pip install -e .
  ```

- Slow first run due to model downloads  
  Sentence-Transformers will download weights on first use. Subsequent runs are faster.

- Memory constraints  
  Use a smaller embedding model, increase `--min-cluster-size`, or reduce UMAP `--n-components`.

## Acknowledgments

Built with [BERTopic](https://github.com/MaartenGr/BERTopic), [sentence-transformers](https://www.sbert.net/), [UMAP](https://umap-learn.readthedocs.io/), and [HDBSCAN](https://hdbscan.readthedocs.io/).
