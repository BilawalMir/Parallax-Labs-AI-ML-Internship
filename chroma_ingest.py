"""
chroma_ingest.py
Set up a persistent ChromaDB collection and ingest chunk records + embeddings.

Usage:
    python chroma_ingest.py
"""

import chromadb
import pandas as pd

from chunking import chunk_dataframe
from embeddings import load_model, embed_chunks, log_performance

DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
DATASET_PATH = "processed_dataset.csv"


def get_collection(db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
    """Get or create a persistent Chroma collection."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def ingest_dataframe(chunks_df: pd.DataFrame, collection, model=None, batch_size: int = 256):
    """
    Embed and ingest chunks from a chunk_dataframe() result into the Chroma
    collection. Processes in batches so large datasets don't blow up memory
    or hit request-size limits.

    chunks_df columns: chunk_id, doc_id, category, chunk_index, chunk_text,
    chunk_length, num_chunks_in_doc (same fields as chunk_document(), just
    as DataFrame rows instead of dicts).
    """
    if chunks_df is None or chunks_df.empty:
        print("No chunks to ingest (empty dataframe).")
        return

    total = len(chunks_df)
    print(f"Ingesting {total} chunks in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        batch = chunks_df.iloc[start:start + batch_size]

        ids = batch["chunk_id"].astype(str).tolist()
        texts = batch["chunk_text"].astype(str).tolist()
        metadatas = batch[
            ["doc_id", "category", "chunk_index", "chunk_length", "num_chunks_in_doc"]
        ].to_dict(orient="records")

        embeddings, stats = embed_chunks(texts, model=model)
        log_performance(stats, note=f"ingest batch {start}-{start + len(batch)}")

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"  Ingested rows {start}-{start + len(batch)} of {total}")

    print(f"Done. Collection '{collection.name}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows.")

    # chunk_dataframe(df, text_column='clean_text', doc_id_column='doc_id',
    #                  category_column='category', chunk_size=1000, chunk_overlap=100)
    chunks_df = chunk_dataframe(df)
    print(f"Produced {len(chunks_df)} chunks from {len(df)} documents.")

    model = load_model()
    client, collection = get_collection()

    ingest_dataframe(chunks_df, collection, model=model)