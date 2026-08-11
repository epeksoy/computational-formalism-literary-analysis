import os
import glob
import json
import nltk
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")                  # txt files here
EMB_DIR = os.path.join(BASE_DIR, "embeddings")                 # per-novel .npy
META_FILE_EMB = os.path.join(BASE_DIR, "processed_files.json")  # embedding resume log
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_WORDS = 200          # drop to 250 if needed
BATCH_SIZE = 16            # reduce if you hit OOM

# Ensure nltk punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)


def chunk_text(text, max_words=300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()

        # handle ultra-long sentences
        if len(words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue

        if current_word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = len(words)
        else:
            current_chunk.append(sentence)
            current_word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_processed():
    if os.path.exists(META_FILE_EMB):
        with open(META_FILE_EMB, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed(processed_set):
    with open(META_FILE_EMB, "w", encoding="utf-8") as f:
        json.dump(sorted(list(processed_set)), f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("   STAGE 1: GENERATE SBERT EMBEDDINGS")
    print("=" * 60)

    # Optional: Google Drive mounting (kept for Colab compatibility)
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
    except ImportError:
        print("Not running in Google Colab environment. Skipping Drive mounting.")

    ensure_dirs()

    # Device / Model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 384  # keep truncation explicit

    if not os.path.exists(CORPUS_DIR):
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    txt_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {CORPUS_DIR}")

    if os.path.exists(METADATA_FILE):
        import pandas as pd
        metadata = pd.read_csv(METADATA_FILE)
        valid_ids = set(metadata['gutenberg_id'].astype(str))
        txt_files = [f for f in txt_files if os.path.splitext(os.path.basename(f))[0] in valid_ids]
        print(f"Filtered corpus to {len(txt_files)} files based on {METADATA_FILE}")

    processed = load_processed()
    print(f"Total files to process: {len(txt_files)} | Already processed: {len(processed)}")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        text_id = os.path.splitext(filename)[0]
        emb_path = os.path.join(EMB_DIR, f"{text_id}.npy")

        # Double check both the json log and actual file existence on disk
        if filename in processed and os.path.exists(emb_path):
            print(f"[SKIP] {filename} already embedded.")
            continue

        print(f"\nProcessing {filename} ...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            print("  -> encoding fallback cp1252")
            with open(file_path, "r", encoding="cp1252", errors="ignore") as f:
                text = f.read()

        chunks = chunk_text(text, max_words=CHUNK_WORDS)
        if len(chunks) < 2:
            print(f"  -> Skipping, not enough chunks (found {len(chunks)})")
            processed.add(filename)
            save_processed(processed)
            continue

        print(f"  -> Computing SBERT embeddings for {len(chunks)} chunks ...")
        embeddings = model.encode(
            chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        np.save(emb_path, embeddings)
        print(f"  -> Saved embeddings to {emb_path}")

        processed.add(filename)
        save_processed(processed)

    print("\nEmbeddings generation completed successfully!")
    print(f"All embeddings are stored in: {EMB_DIR}")


if __name__ == "__main__":
    main()
