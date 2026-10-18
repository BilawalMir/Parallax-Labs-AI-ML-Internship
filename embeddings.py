"""
embeddings.py
Generate sentence-transformer embeddings for text chunks and log performance.

Usage:
    python embeddings.py
"""

import csv
import os
import time
from datetime import datetime

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # fast + solid quality, 384-dim vectors
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "embedding_perf.csv")


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and return a SentenceTransformer model."""
    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model


def embed_chunks(chunks: list[str], model: SentenceTransformer = None):
    """
    Embed a list of chunk texts.

    Returns:
        embeddings: numpy array of shape (num_chunks, embedding_dim)
        stats: dict with timing info
    """
    if not chunks:
        return [], {
            "num_chunks": 0,
            "total_time_sec": 0.0,
            "avg_time_per_chunk_sec": 0.0,
            "embedding_dim": 0,
        }

    if model is None:
        model = load_model()

    start = time.perf_counter()
    embeddings = model.encode(chunks, show_progress_bar=True)
    elapsed = time.perf_counter() - start

    stats = {
        "num_chunks": len(chunks),
        "total_time_sec": round(elapsed, 4),
        "avg_time_per_chunk_sec": round(elapsed / len(chunks), 6),
        "embedding_dim": embeddings.shape[1],
    }
    return embeddings, stats


def log_performance(stats: dict, model_name: str = MODEL_NAME, note: str = ""):
    """Append a row of timing stats to logs/embedding_perf.csv."""
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "model_name", "num_chunks",
                "total_time_sec", "avg_time_per_chunk_sec",
                "embedding_dim", "note"
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            model_name,
            stats["num_chunks"],
            stats["total_time_sec"],
            stats["avg_time_per_chunk_sec"],
            stats["embedding_dim"],
            note,
        ])
    print(f"Logged performance to {LOG_FILE}")


if __name__ == "__main__":
    from chunking import chunk_document

    sample_text = """
    Replace this with a real sample document, or load one from disk, e.g.:
    with open("sample_doc.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()
    This placeholder is just long enough to produce a couple of chunks so the
    script runs end-to-end without extra setup. In practice you should load a
    real document relevant to your project so the timing numbers reflect
    realistic chunk sizes and counts. Add more filler text here as needed to
    simulate a longer document if you want to test multi-chunk behavior.
    """

    # chunk_document(doc_id, category, text, chunk_size=1000, chunk_overlap=100)
    records = chunk_document("doc1", "sample", sample_text)
    chunk_texts = [r["chunk_text"] for r in records]

    model = load_model()
    embeddings, stats = embed_chunks(chunk_texts, model=model)

    print(f"Embedded {stats['num_chunks']} chunks in {stats['total_time_sec']}s "
          f"({stats['avg_time_per_chunk_sec']*1000:.2f} ms/chunk), "
          f"dim={stats['embedding_dim']}")

    log_performance(stats, note="sample run")