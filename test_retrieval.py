"""
test_retrieval.py
Run a set of test queries against the ChromaDB collection, log latency
per query, and print the retrieved results.

Usage:
    python test_retrieval.py
"""

import csv
import os
import time
from datetime import datetime

from chroma_ingest import get_collection
from embeddings import load_model

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "retrieval_perf.csv")

# Domain-relevant queries reflecting actual topics in the 20 Newsgroups
# dataset (sci.space, sci.med, comp.security/sci.crypt, writing-related
# threads), so retrieval results show real semantic relevance rather than
# generic/synthetic phrasing.
TEST_QUERIES = [
    "how do spacecraft communicate with mission control over long distances",
    "what causes epileptic seizures and how are they treated",
    "best practices for creating a strong password",
    "how does encryption keep data secure",
    "tips for improving your writing style",
]


def search(query: str, model, collection, top_k: int = 5):
    """Embed a query, run it against the collection, and time the round trip."""
    start = time.perf_counter()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    latency = time.perf_counter() - start
    return results, latency


def log_latency(query: str, latency: float, num_results: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "latency_sec", "num_results"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            round(latency, 5),
            num_results,
        ])


def run_test_queries(queries: list[str] = TEST_QUERIES, top_k: int = 5):
    model = load_model()
    _, collection = get_collection()

    if collection.count() == 0:
        print("Collection is empty. Run chroma_ingest.py first to add data.")
        return

    for query in queries:
        results, latency = search(query, model, collection, top_k=top_k)
        num_results = len(results.get("documents", [[]])[0])

        print(f"\nQuery: {query!r}")
        print(f"Latency: {latency*1000:.2f} ms | Results: {num_results}")
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            preview = doc[:100].replace("\n", " ")
            print(f"  [{i}] (distance={distance:.4f}) {preview}...")

        log_latency(query, latency, num_results)

    print(f"\nAll latencies logged to {LOG_FILE}")


if __name__ == "__main__":
    run_test_queries()