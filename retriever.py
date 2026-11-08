"""
retriever.py
------------
Retrieval layer used by generate_answer.py.

UPDATE (post Task 2 wiring)
==============================
Task 2 (`chunking.py`, `embeddings.py`, `chroma_ingest.py`) turned out to
already exist on `main` -- it was built earlier and just hadn't been checked
against this file. Two things needed to change from the original best-guess
version of this module:

  1. Collection name: `chroma_ingest.py` creates/writes to a collection named
     `"documents"` (`COLLECTION_NAME` in that file), not `"parallax_docs"`.
     Confirmed by the runtime error this file used to raise:
     "Could not open Chroma collection 'parallax_docs' ... does not exist."
  2. Embedding function: `chroma_ingest.py` embeds chunks explicitly with
     sentence-transformers' `all-MiniLM-L6-v2` (via `embeddings.load_model()`)
     and stores those vectors directly with `collection.upsert(embeddings=...)`.
     The original version of this file queried with `query_texts=[query]`,
     which makes Chroma embed the query using its own default embedding
     function instead -- a different embedding space than the one the chunks
     were actually stored in. That wouldn't error out, it would just quietly
     return poor-quality matches. Fixed by loading the same
     `all-MiniLM-L6-v2` model here and querying with `query_embeddings=`.

`ChromaRetriever` now matches `chroma_ingest.py`'s contract exactly.
`TfidfRetriever` is kept as-is: a dependency-light fallback so the pipeline
still runs end-to-end even without `chroma_db/` present.

`get_retriever()` at the bottom picks Chroma automatically if `chroma_db/`
exists, otherwise falls back to TF-IDF. Everything downstream only depends on
`.retrieve(query, top_k) -> list[RetrievedChunk]`, so no changes are needed
in generate_answer.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    """A single chunk returned by either retriever backend.

    Both `ChromaRetriever` and `TfidfRetriever` return lists of this same
    type so `generate_answer.py` can consume either backend without caring
    which one produced the results.

    Attributes:
        text: The chunk's raw text content.
        score: Similarity score in [0, 1] (higher = more relevant).
            Chroma distances are converted to this scale; TF-IDF cosine
            similarity is already in this range.
        doc_id: Chroma's chunk ID for Chroma results, or a numeric-string
            row index for TF-IDF results.
        metadata: topic_label/sentiment_label/etc. for Chroma results;
            always `{}` for TF-IDF results, which carry no metadata.
    """

    text: str
    score: float
    doc_id: str
    metadata: dict = None

    def __post_init__(self):
        """Normalize `metadata=None` to an empty dict so callers can always
        do `chunk.metadata.get(...)` without a None-check."""
        if self.metadata is None:
            self.metadata = {}


# --------------------------------------------------------------------------
# Option 1: Chroma-based retriever (matches the .gitignore-implied Week 2 plan)
# --------------------------------------------------------------------------

CHROMA_DB_PATH = "chroma_db"
# Matches COLLECTION_NAME in chroma_ingest.py -- override with the
# CHROMA_COLLECTION_NAME env var if that ever changes.
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
# Matches MODEL_NAME in embeddings.py -- must stay in sync so queries land in
# the same embedding space as the stored chunk vectors.
EMBEDDING_MODEL_NAME = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class ChromaRetriever:
    """Queries an existing persistent Chroma collection built by chroma_ingest.py."""

    def __init__(self, persist_path: str = CHROMA_DB_PATH, collection_name: str = CHROMA_COLLECTION_NAME):
        """Open the persistent Chroma collection and load the embedding model.

        Args:
            persist_path: Directory where the Chroma store lives on disk
                (must match the path `chroma_ingest.py` wrote to).
            collection_name: Name of the collection to open (must match
                `COLLECTION_NAME` in `chroma_ingest.py`).

        Raises:
            ImportError: If `chromadb` or `sentence-transformers` is not
                installed.
            ValueError: If the collection can't be opened (wrong name, or
                `chroma_ingest.py` hasn't been run yet).
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. Run `pip install chromadb` (and add it "
                "to requirements.txt), or use TfidfRetriever instead."
            ) from e

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` (matches embeddings.py's "
                "dependency), or use TfidfRetriever instead."
            ) from e

        self._client = chromadb.PersistentClient(path=persist_path)
        try:
            # No embedding_function= here on purpose: chroma_ingest.py stores
            # pre-computed embeddings directly (collection.upsert(embeddings=...)),
            # it doesn't rely on Chroma's built-in embedding function. We embed
            # queries ourselves below with the same model, for the same reason.
            self._collection = self._client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(
                f"Could not open Chroma collection '{collection_name}' at "
                f"'{persist_path}': {e}. Check CHROMA_COLLECTION_NAME matches "
                f"the name used in chroma_ingest.py."
            ) from e

        # Load once at construction, not per-query -- same idea as
        # TfidfRetriever building its vectorizer once in __init__.
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 4, where: dict | None = None) -> list[RetrievedChunk]:
        """Embed `query` and return the top-k nearest chunks from Chroma.

        Args:
            query: Natural-language query string. An empty/whitespace-only
                query short-circuits to `[]` without calling Chroma.
            top_k: Maximum number of chunks to return.
            where: Optional Chroma metadata filter, e.g.
                `{"topic_label": "space_shuttle"}` or
                `{"$and": [{"topic_label": "..."}, {"sentiment_label": "..."}]}`
                (see `verify_filter.py` for the filter shapes this
                collection supports).

        Returns:
            Up to `top_k` `RetrievedChunk` objects, ordered by descending
            similarity score.
        """
        if not query or not query.strip():
            return []
        query_embedding = self._model.encode([query]).tolist()
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{}] * len(ids)
        chunks = []
        for cid, doc, dist, meta in zip(ids, docs, distances, metadatas):
            # Chroma returns a distance (lower = more similar); convert to a
            # similarity-style score in [0, 1] so it's comparable to the
            # TF-IDF retriever's cosine scores for the out-of-domain gate.
            score = 1.0 / (1.0 + dist)
            chunks.append(RetrievedChunk(text=doc, score=score, doc_id=cid, metadata=meta or {}))
        return chunks


# --------------------------------------------------------------------------
# Option 2: TF-IDF fallback (no vector DB required)
# --------------------------------------------------------------------------

class TfidfRetriever:
    """Tiny TF-IDF retriever. Runnable today with zero extra infrastructure."""

    def __init__(self, dataset_path: str = "processed_dataset.csv", text_col: str = "text"):
        """Load the dataset and fit a TF-IDF vectorizer over its text column.

        Args:
            dataset_path: Path to the cleaned CSV produced by Week 1's
                `generate_clean_dataset.py`.
            text_col: Name of the column containing the document text.

        Raises:
            FileNotFoundError: If `dataset_path` doesn't exist.
            ValueError: If `text_col` is missing, or no rows remain after
                dropping blank text rows.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Could not find '{dataset_path}'. Run Week 1's "
                f"generate_clean_dataset.py first, or point --dataset at your "
                f"own processed CSV with a '{text_col}' column."
            )
        df = pd.read_csv(dataset_path)
        if text_col not in df.columns:
            raise ValueError(f"'{dataset_path}' has no '{text_col}' column.")
        # drop empty rows defensively -- Week 1's cleaning can produce a few
        df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"'{dataset_path}' has no usable rows after filtering.")

        self.texts = df[text_col].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 4, where: dict | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks by TF-IDF cosine similarity to `query`.

        Args:
            query: Natural-language query string. An empty/whitespace-only
                query short-circuits to `[]`.
            top_k: Maximum number of chunks to return.
            where: Not supported by this backend; passing a truthy value
                raises, since TF-IDF results carry no metadata to filter on.

        Returns:
            Up to `top_k` `RetrievedChunk` objects with positive similarity,
            ordered by descending score. Zero-similarity matches are dropped.

        Raises:
            ValueError: If `where` is given (metadata filtering needs the
                Chroma backend).
        """
        if where:
            raise ValueError(
                "TfidfRetriever has no metadata (topic/sentiment labels) to filter on -- "
                "metadata filtering requires the Chroma backend (run chroma_ingest.py, "
                "topic_modeling.py, and sentiment_analysis.py first)."
            )
        if not query or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(text=self.texts[i], score=float(sims[i]), doc_id=str(i))
            for i in top_idx
            if sims[i] > 0  # drop zero-similarity results outright
        ]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def get_retriever(dataset_path: str = "processed_dataset.csv"):
    """
    Picks Chroma automatically if a `chroma_db/` store exists locally
    (i.e. chroma_ingest.py has been run), otherwise falls back to the
    TF-IDF retriever over `dataset_path` so the pipeline always runs.

    generate_answer.py only calls `.retrieve(query, top_k)` and reads
    `.text` / `.score` off the results, so either branch works transparently.
    """
    if os.path.isdir(CHROMA_DB_PATH):
        try:
            return ChromaRetriever()
        except Exception as e:
            print(f"[warn] Found '{CHROMA_DB_PATH}/' but couldn't open it as a Chroma "
                  f"retriever ({e}). Falling back to TF-IDF over '{dataset_path}'.")
    return TfidfRetriever(dataset_path=dataset_path)
