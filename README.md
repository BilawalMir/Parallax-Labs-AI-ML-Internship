# Parallax Labs — ML Internship

A single repository will be used for all Parallax Labs internship submissions, with updates made week by week rather than creating a new repository for each task.

**Student:** Bilawal Ijaz 

---

## Progress Log

### Task 1 — Environment Setup & Dataset Cleaning Pipeline

**Status: complete.**

**What's done:**
- `requirements.txt` finalized with all libraries needed for dataset acquisition,
  validation, and text cleaning
- `verify_setup.py` — verification script that imports every dependency **and**
  runs a small functional check for each (not just `import x`), so a broken install
  is caught immediately instead of failing later mid-pipeline
- `load_and_validate_dataset.py` — acquires the 20 Newsgroups dataset (~18,846 docs,
  headers/footers/quotes intentionally kept as real-world noise) and validates it:
  null checks, empty/whitespace checks, duplicate checks, encoding checks, and
  document-length distribution. Outputs `data_raw.csv` and `data_validation_summary.md`.
- `text_cleaning.py` — modular, independently-testable cleaning functions:
  `fix_encoding`, `remove_newsgroup_headers`, `remove_quoted_lines`,
  `remove_signature`, `remove_html_tags`, `remove_urls`, `remove_email_addresses`,
  `remove_emoji`, `normalize_whitespace`, `detect_language`, and a composed
  `clean_text()` pipeline. Every function returns `""` (or `"unknown"` for
  language detection) instead of crashing on `None`/empty/whitespace-only input.
- `test_text_cleaning.py` — 42 unit tests covering every function, including
  edge cases (None, empty, whitespace-only, non-string input, mixed-language
  text, idempotency). All 42 passing.
- `generate_clean_dataset.py` — applies the full cleaning pipeline + language
  detection to every document in `data_raw.csv`, and outputs:
  - `processed_dataset.csv` — final clean dataset, ready for chunking/embedding
  - `data_quality_report.md` — before/after quality report (length reduction,
    docs that became empty after cleaning, language distribution, post-clean
    duplicates)

**Task 1 status: complete.** All 4 stages done — environment, dataset
acquisition/validation, cleaning functions, unit tests + final clean dataset.

**Note:** repo uses a flat file structure (no subfolders) for simplicity.

### Task 2 — Chunking, Embeddings & Vector DB

**Status: complete.**

**What's done:**

- `chunking.py` — recursive character text splitting (custom implementation,
  same approach as LangChain's `RecursiveCharacterTextSplitter`), chunk
  size 1000 / overlap 100. Produces 35,072 chunks from `processed_dataset.csv`.
- `test_chunking.py` — unit tests covering edge cases (None/empty input,
  overlap >= chunk size, paragraph-boundary splitting, hard character-split
  fallback, unique chunk IDs).
- `embeddings.py` — generates embeddings with `sentence-transformers`
  (`all-MiniLM-L6-v2`, 384-dim) and logs timing to `logs/embedding_perf.csv`.
- `chroma_ingest.py` — sets up a ChromaDB collection (`"documents"`) and
  ingests all chunks + embeddings.
- `test_retrieval.py` — runs domain-relevant semantic search queries
  against the collection and logs latency to `logs/retrieval_perf.csv`.


**Running Task 2** (after Task 1's `processed_dataset.csv` exists):

```bash
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
```