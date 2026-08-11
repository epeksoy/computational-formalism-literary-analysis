# Project Gutenberg Narrative Arc Analysis

This repository provides a pipeline to download books from Project Gutenberg, compute their semantic narrative arcs using SBERT (Sentence-BERT), cluster them, and validate the extracted narrative shapes. 

## Workflow Scripts

1. **`0. gutenberg_download.py`**: A utility to fetch text from Project Gutenberg using book IDs. It automatically strips headers/footers to produce clean text files.
2. **`1. generate_sbert_embeddings.py`**: Tokenizes and chunks the raw text files. Passes the chunks through `all-mpnet-base-v2` to generate semantic embeddings for each book.
3. **`2. analyze_and_plot_clusters.py`**: Averages paragraph embeddings across the timeline of each novel to trace narrative arcs, then clusters the shapes across the corpus using hierarchical/SVD methods.
4. **`3. plot_individual_arcs.py`**: Helper script to generate visualizations for specifically requested novels or authors.
5. **`4. validate_clusters.py`**: Conducts Monte Carlo significance testing and evaluates cluster compactness (Silhouette, Calinski-Harabasz, Davies-Bouldin metrics) to validate narrative archetypes.

## Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Note on Datasets
The large dataset outputs (corpus text files, numpy embedding matrices, and large Parquet files) are omitted from this repository due to their size. Ensure you run the download script and embedding generator locally to reproduce the data.
