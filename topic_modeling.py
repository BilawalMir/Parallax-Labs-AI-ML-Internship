"""
topic_modeling.py
------------------
Week 4, deliverable 1: discover corpus themes with topic modeling and write
the results back into the Chroma vector DB as filterable metadata.

Tries BERTopic first (embedding-based clustering -> usually more coherent
topics on short, noisy chunks like ours). If BERTopic (or its heavier deps:
umap-learn, hdbscan, sentence-transformers) isn't installed, falls back
automatically to sklearn LDA so this still runs on a minimal environment.
Force one explicitly with --method bertopic / --method lda.

Edge cases (matches the "too_short" / "garbled" triage already used in
ner_enrichment.py, via nlp_utils.is_usable_for_nlp):
  - too_short chunks  -> topic_id = -1, topic_label = "too_short_to_model"
  - garbled chunks    -> topic_id = -2, topic_label = "garbled_skipped"
  - everything else goes into the model.

Usage:
    python topic_modeling.py
    python topic_modeling.py --method lda --n-topics 20
    python topic_modeling.py --db ./chroma_db --collection documents
"""

import argparse
import csv
import os
import time
from datetime import datetime

import chromadb

from nlp_utils import is_usable_for_nlp, truncate_for_preview

BATCH_PRINT = 1000
UPDATE_BATCH = 500


def fetch_all_chunks(collection, batch_print=BATCH_PRINT):
    """Paginate through the whole collection, same pattern as ner_enrichment.py."""
    print(f"Loading chunks from Chroma collection '{collection.name}' ...")
    ids, docs, metas = [], [], []
    offset = 0
    page_size = 1000
    total = collection.count()
    while offset < total:
        res = collection.get(
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids.extend(res["ids"])
        docs.extend(res["documents"])
        metas.extend(res["metadatas"])
        offset += len(res["ids"])
        if offset % batch_print == 0 or offset >= total:
            print(f"  fetched {min(offset, total)}/{total} chunks...")
        if len(res["ids"]) == 0:
            break
    print(f"Loaded {len(ids)} chunks.")
    return ids, docs, metas


def triage(ids, docs):
    """Split ids/docs into usable vs. too_short vs. garbled, preserving order info."""
    usable_ids, usable_docs = [], []
    too_short_ids, garbled_ids = [], []
    for _id, doc in zip(ids, docs):
        status = is_usable_for_nlp(doc)
        if status == "usable":
            usable_ids.append(_id)
            usable_docs.append(doc)
        elif status == "too_short":
            too_short_ids.append(_id)
        else:
            garbled_ids.append(_id)
    print(
        f"{len(usable_ids)} chunks usable for topic modeling, "
        f"{len(too_short_ids)} too short, {len(garbled_ids)} garbled -- skipped from model fit."
    )
    return usable_ids, usable_docs, too_short_ids, garbled_ids


def run_bertopic(docs, n_topics):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    kwargs = {"vectorizer_model": vectorizer, "verbose": True, "calculate_probabilities": False}
    if n_topics:
        kwargs["nr_topics"] = n_topics
    model = BERTopic(**kwargs)
    assignments, _ = model.fit_transform(docs)

    topic_info = {}
    for topic_id in set(assignments):
        words = [w for w, _ in model.get_topic(topic_id)] if topic_id != -1 else []
        label = "outlier" if topic_id == -1 else "_".join(words[:3]) if words else f"topic_{topic_id}"
        topic_info[topic_id] = {"keywords": ", ".join(words[:10]), "label": label}
    return assignments, topic_info


def run_lda(docs, n_topics):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    n_topics = n_topics or 15
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9, ngram_range=(1, 1))
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="online")
    doc_topic = lda.fit_transform(X)
    assignments = doc_topic.argmax(axis=1).tolist()

    feature_names = vectorizer.get_feature_names_out()
    topic_info = {}
    for topic_id, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:10]
        words = [feature_names[i] for i in top_idx]
        topic_info[topic_id] = {"keywords": ", ".join(words), "label": "_".join(words[:3])}
    return assignments, topic_info


def update_metadata(collection, ids, metadata_updates, update_batch=UPDATE_BATCH):
    for i in range(0, len(ids), update_batch):
        batch_ids = ids[i:i + update_batch]
        batch_meta = metadata_updates[i:i + update_batch]
        collection.update(ids=batch_ids, metadatas=batch_meta)


def write_topic_summary(topic_info, assignments, ids, docs, path="topic_summary.md"):
    """Human-readable summary for manual validation of topic quality."""
    from collections import Counter, defaultdict

    counts = Counter(assignments)
    examples = defaultdict(list)
    for _id, doc, topic_id in zip(ids, docs, assignments):
        if len(examples[topic_id]) < 3:
            examples[topic_id].append(truncate_for_preview(doc, 160))

    lines = ["# Topic Summary (for manual validation)\n"]
    lines.append(
        "Review each topic below: do the keywords and example snippets form a "
        "coherent theme? Note anything incoherent, duplicated, or too broad in "
        "NLP_METADATA_REPORT.md.\n"
    )
    for topic_id in sorted(topic_info.keys(), key=lambda t: -counts.get(t, 0)):
        info = topic_info[topic_id]
        lines.append(f"## Topic {topic_id} — `{info['label']}`  ({counts.get(topic_id, 0)} chunks)")
        lines.append(f"**Keywords:** {info['keywords']}\n")
        lines.append("**Examples:**")
        for ex in examples[topic_id]:
            lines.append(f"- \"{ex}\"")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path} for manual topic validation.")


def log_performance(path, row, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Logged performance to {path}")


def main():
    parser = argparse.ArgumentParser(description="Topic modeling over the Chroma corpus.")
    parser.add_argument("--db", default="./chroma_db")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--method", choices=["auto", "bertopic", "lda"], default="auto")
    parser.add_argument("--n-topics", type=int, default=None, help="Target topic count (LDA default 15; BERTopic default auto).")
    args = parser.parse_args()

    start = time.time()
    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_collection(args.collection)

    ids, docs, _metas = fetch_all_chunks(collection)
    usable_ids, usable_docs, too_short_ids, garbled_ids = triage(ids, docs)

    method = args.method
    if method == "auto":
        try:
            import bertopic  # noqa: F401
            method = "bertopic"
        except ImportError:
            print("BERTopic not installed -- falling back to LDA. "
                  "(pip install bertopic umap-learn hdbscan sentence-transformers for the richer model.)")
            method = "lda"

    print(f"Running topic model: {method} on {len(usable_docs)} chunks...")
    if method == "bertopic":
        assignments, topic_info = run_bertopic(usable_docs, args.n_topics)
    else:
        assignments, topic_info = run_lda(usable_docs, args.n_topics)

    n_topics_found = len({t for t in topic_info.keys() if t != -1})
    print(f"Found {n_topics_found} topics ({method}).")

    # Build metadata updates for usable chunks
    metadata_updates = []
    for topic_id in assignments:
        info = topic_info[topic_id]
        metadata_updates.append({
            "topic_id": int(topic_id),
            "topic_label": info["label"],
            "topic_keywords": info["keywords"],
        })
    update_metadata(collection, usable_ids, metadata_updates)

    # Edge-case chunks still get a metadata value so filters never hit a missing field
    if too_short_ids:
        update_metadata(collection, too_short_ids,
                         [{"topic_id": -1, "topic_label": "too_short_to_model", "topic_keywords": ""}] * len(too_short_ids))
    if garbled_ids:
        update_metadata(collection, garbled_ids,
                         [{"topic_id": -2, "topic_label": "garbled_skipped", "topic_keywords": ""}] * len(garbled_ids))

    total_updated = len(usable_ids) + len(too_short_ids) + len(garbled_ids)
    print(f"Updated topic metadata for {total_updated} chunks in '{args.collection}'.")

    write_topic_summary(topic_info, assignments, usable_ids, usable_docs)

    elapsed = time.time() - start
    log_performance(
        os.path.join("logs", "topic_modeling_perf.csv"),
        [datetime.now().isoformat(timespec="seconds"), method, len(docs), len(usable_docs),
         n_topics_found, round(elapsed, 1)],
        ["timestamp", "method", "total_chunks", "usable_chunks", "n_topics", "elapsed_seconds"],
    )
    print(f"Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
