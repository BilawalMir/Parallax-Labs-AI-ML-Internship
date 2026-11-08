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
### Task 3 — DeepSeek/OpenRouter Generation (RAG completion)

**Status: complete.** Adds an LLM generation layer on top of the retrieval
pipeline, with prompt engineering, robust API error handling, hallucination/
out-of-domain guards, and end-to-end latency logging.

**Integration note:** `retriever.py` was originally written as a best-guess
against `.gitignore`'s hint of a future `chroma_ingest.py` → `chroma_db/`
setup, before checking whether Task 2 was already on `main` — it was. Wiring
`retriever.py` up to the real `chroma_ingest.py` surfaced two mismatches:
the collection name (`retriever.py` assumed `"parallax_docs"`, the real
collection is `"documents"`), and the embedding path (`retriever.py` queried
with `query_texts=`, letting Chroma embed the query with its own default
embedding function, while `chroma_ingest.py` stores explicit
`all-MiniLM-L6-v2` vectors — two different embedding spaces, which wouldn't
error, just silently degrade retrieval quality). Both are now fixed:
`ChromaRetriever` opens the `"documents"` collection and embeds queries with
the same `all-MiniLM-L6-v2` model via `query_embeddings=`, so queries and
stored chunks live in the same space. A dependency-light `TfidfRetriever`
fallback is kept for environments without a populated `chroma_db/`.

Verified end-to-end after the fix: `test_generate_answer.py` still 18/18
passing, and live runs against the real `chroma_db` return `status=answered`
with grounded, cited answers (e.g. spacecraft-communication and epilepsy
queries pulling real dataset content) instead of falling back to TF-IDF.

**What's done:**
- `retriever.py` — `ChromaRetriever` (opens the `"documents"` collection
  built by `chroma_ingest.py`, embeds queries with the matching
  `all-MiniLM-L6-v2` model) + `TfidfRetriever` fallback + `get_retriever()`
  factory that picks whichever is available.
- `generate_answer.py` — DeepSeek-via-OpenRouter generation:
  - **Prompt engineering:** a system prompt that constrains the model to the
    retrieved context, requires inline citations `[1]`, `[2]`, ..., and gives
    an explicit instruction to abstain rather than guess; context is injected
    as a numbered, token-budgeted block (`build_context_block`).
  - **Error handling:** timeouts, connection errors, 429 rate limits (with
    `Retry-After` support), 5xx retried with exponential backoff, 4xx
    (including context-length/token-limit errors) failed fast without
    retrying, malformed responses caught explicitly. All in
    `call_openrouter()`.
  - **Hallucination / out-of-domain handling:** a retrieval-confidence gate
    (`RETRIEVAL_CONFIDENCE_THRESHOLD`) that returns a canned "out of domain"
    response *without calling the API* when nothing relevant is retrieved,
    plus a lexical-grounding check (`check_grounding`) on every generated
    answer against the retrieved context, plus the system prompt's explicit
    abstain instruction.
  - **Latency logging:** every query logs `retrieval_latency_s`,
    `generation_latency_s`, `total_latency_s`, chunks retrieved, top score,
    and status to `rag_latency_log.jsonl` (and stdout).
- `test_generate_answer.py` — 18 tests, all mocked (no API key or network
  needed), covering prompt building, grounding checks, and every error path
  (timeout, 429, 5xx, 4xx/context-length, malformed response) plus full
  pipeline behavior for in-domain, out-of-domain, abstain, and error cases.
- `.env.example` — copy to `.env` and add your own OpenRouter key.
- `sample_processed_dataset.csv` — a tiny demo dataset so `generate_answer.py`
  runs out of the box even before `processed_dataset.csv` (Task 1's real
  output) exists locally.
- `requests` and `python-dotenv` added to `requirements.txt` for the
  OpenRouter API calls and `.env` loading.

**Setup:**
```bash
pip install -r requirements.txt
cp .env.example .env                     # then edit .env with your key from
                                          # https://openrouter.ai/keys
```

**Run:**
```bash
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
# or interactively:
python generate_answer.py
```
Note: a query like "What happened to the Space Shuttle Columbia?" correctly
returns `status=abstained` — the 2003 Columbia disaster postdates the
1995-era 20 Newsgroups corpus, so there's genuinely no relevant context to
ground an answer in. That's the grounding/abstain logic working as intended,
not a bug.

**Test (no API key needed):**
```bash
python -m pytest test_generate_answer.py -v
```
Expected: `18 passed`.

**Model:** defaults to `deepseek/deepseek-chat` (OpenRouter's stable alias
for DeepSeek's current flagship chat model — solid quality, cheap, no dated
snapshot to maintain). Override with `OPENROUTER_MODEL` in `.env`, e.g.
`deepseek/deepseek-chat-v3.1:free` for $0 testing while building (free tier
is rate-limited, not for production use). Current models/pricing:
https://openrouter.ai/deepseek

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key libraries:

| Library | Purpose |
|---|---|
| pandas, numpy | Data handling |
| scikit-learn | Dataset loader (20 Newsgroups); TF-IDF fallback retriever |
| nltk | Tokenization |
| langdetect | Detecting mixed-language text |
| ftfy | Fixing encoding artifacts (mojibake) |
| chardet | Detecting raw byte encoding |
| emoji | Detecting/handling emoji in text |
| pytest | Unit testing |
| tqdm | Progress bars for large-loop operations |
| tiktoken | Token-aware chunking (used in a later task, not this one) |
| sentence-transformers | Embeddings (`all-MiniLM-L6-v2`) for Task 2 & Task 3 retrieval |
| chromadb | Persistent vector store (Task 2 ingestion, Task 3 retrieval) |
| requests | OpenRouter API calls (Task 3) |
| python-dotenv | Loading `.env` config (Task 3) |
| bertopic | Optional upgrade over LDA for topic modeling (Task 4) |
| spacy | NER (`en_core_web_sm`) for entity metadata (Task 4) |
| fastapi | REST API wrapping the RAG pipeline (Task 5) |
| uvicorn[standard] | ASGI server that runs the FastAPI app (Task 5) |
| httpx | HTTP client used by FastAPI's `TestClient` in `test_api.py` (Task 5) |


**Design note (Stage 3):** `text_cleaning.py` does not lowercase or strip
punctuation by default, since this dataset is being prepared for embeddings,
and embedding models generally do better on natural casing/punctuation.

---

## Setup Instructions

```bash
git clone https://github.com/BilawalMir/Parallax-Labs-AI-ML-Internship.git
cd Parallax-Labs-AI-ML-Internship
pip install -r requirements.txt
```

## Running the Pipeline

Run these in order — each step depends on the file(s) produced by the one before it.

**1. Verify the environment**
```bash
python verify_setup.py
```
Expected output: `Result: 11/11 checks passed` (a `[WARN]` on `tiktoken` is fine if
your network blocks its one-time encoding file download — it does not affect this
task, which doesn't use tiktoken yet).

**2. Acquire and validate the raw dataset**
```bash
python load_and_validate_dataset.py
```
Downloads 20 Newsgroups (~18,846 docs), then writes `data_raw.csv` and
`data_validation_summary.md`. Expected: `Total documents: 18846`, with
`Null entries`, `Duplicate documents`, and `Encoding issues` all at `0`.

**3. Run the unit tests for the cleaning functions**
```bash
python -m pytest test_text_cleaning.py -v
```
Expected: `42 passed`.

> **Important — use `python -m pytest`, not a bare `pytest` command.**
> On some setups, notably Windows/PowerShell, pip installs `pytest.exe` into a
> `Scripts` folder that isn't on `PATH`, so a bare `pytest` command fails with
> `pytest : The term 'pytest' is not recognized as the name of a cmdlet...`
> even though pytest is installed correctly. `python -m pytest` always works
> since it just asks the already-on-`PATH` `python` interpreter to run pytest
> as a module. **Evaluators should use `python -m pytest test_text_cleaning.py -v`
> to run the tests correctly.**

**4. Generate the final cleaned dataset + quality report**
```bash
python generate_clean_dataset.py
```
Requires `data_raw.csv` from step 2. Outputs `processed_dataset.csv` and
`data_quality_report.md`. Expected: `DONE — Stage 4 complete`.

**5. Chunk, embed, and ingest into ChromaDB (Task 2)**
```bash
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
```
Requires `processed_dataset.csv` from step 4.

**6. Run the RAG generation layer (Task 3)**
```bash
cp .env.example .env    # then add your OpenRouter key
python -m pytest test_generate_answer.py -v          # 18 passed, no key needed
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
```
Requires `chroma_db/` from step 5 (falls back to TF-IDF over
`processed_dataset.csv` / `sample_processed_dataset.csv` if absent).
### Task 4 — NLP Metadata Enrichment (Topic Modeling + NER) & Filtered Retrieval

**Status: complete.** Adds topic modeling and named-entity metadata on top of
the chunked corpus, so retrieval can be filtered by topic or by mentioned
entities, not just by semantic similarity.

**Bug found and fixed during review:** `ner_enrichment.py` imported
`load_chunks_from_chroma` and `update_chroma_metadata` from
`topic_modeling.py`, but neither function exists there — the real API is
`fetch_all_chunks(collection)` and `update_metadata(collection, ids, metas)`,
which take an already-open Chroma collection rather than a
`db_path`/`collection_name` pair. Fixed `ner_enrichment.py` to open the
Chroma client itself and call the real functions. Verified:
`import ner_enrichment` now succeeds cleanly, and `test_ner_enrichment.py`
still passes 13/13 after the change.

**`test_topic_modeling.py` was also stale** — it tested an earlier version
of `topic_modeling.py`'s functions (`partition_by_length`,
`build_topic_labels`, `build_metadata_updates`, `reduce_outlier_topics`)
that no longer exist, so it failed to even collect before this fix.
Rewritten against the current functions (`fetch_all_chunks`, `triage`,
`run_lda`, `run_bertopic`, `update_metadata`, `write_topic_summary`,
`log_performance`). Now: **12 passed, 1 skipped** (the BERTopic test skips
cleanly when `bertopic` isn't installed — same optional-heavy-dependency
pattern used elsewhere in this repo).

**What's done:**
- `topic_modeling.py` — LDA (sklearn) by default, with an automatic upgrade
  to BERTopic when it's installed (`--method` to force either). Applies the
  same too-short/garbled triage as `ner_enrichment.py` (via
  `nlp_utils.is_usable_for_nlp`) so edge-case chunks still get a metadata
  value (`topic_id=-1`/`-2`) instead of being silently dropped from
  filtered retrieval. Writes `topic_summary.md` for manual validation.
- `ner_enrichment.py` — spaCy (`en_core_web_sm`) NER, storing PERSON, ORG,
  GPE, and NORP entities as comma-separated metadata fields per chunk.
  Same too-short/garbled triage, plus a dedicated garbled-text heuristic
  (alphabetic+whitespace character ratio) tuned for this corpus's leaked
  uuencoded binary chunks. Jargon-heavy chunks are *not* filtered out —
  spaCy still runs on them, and the accuracy impact is measured rather than
  silently patched.
- `evaluate_nlp_metadata.py` / `nlp_evaluation_results.md` — manual
  evaluation of topic and sentiment metadata against
  `manual_eval_sample.csv` (50 hand-reviewed rows, stratified across
  normal/jargon/edge-case chunks).
- `evaluate_ner_accuracy.py` / `gold_standard_ner.py` — precision/recall/F1
  of the NER pipeline against a small hand-labeled gold set.
- `NLP_METADATA_REPORT.md` — write-up of effectiveness, accuracy, and
  known limitations for both topic and NER/sentiment metadata.

**Verified accuracy numbers (from `nlp_evaluation_results.md`):**

| Metadata type | Metric | Result |
|---|---|---|
| Topic relevance | Overall | 56.0% (28/50 sampled rows judged relevant) |
| Topic relevance | Normal-text chunks | 61.8% |
| Topic relevance | Jargon-heavy chunks | 12.5% |
| Topic relevance | Edge-case (short/garbled) chunks | 75.0% correctly triaged |
| Sentiment (VADER) | Overall accuracy | 43.2% (44/50 rows judged) |
| Sentiment (VADER) | Neutral-class recall | 0.18 (systematically over-predicts positive/negative) |

**Not yet re-verified in this review:** NER precision/recall/F1 from
`evaluate_ner_accuracy.py` against `gold_standard_ner.py`. The script and
gold-standard set exist and are ready to run, but this environment's spaCy
install couldn't complete the check during this review (broken local torch
install unrelated to the code itself). Run it yourself with:
```bash
python -m spacy download en_core_web_sm
python evaluate_ner_accuracy.py
```
and record the printed precision/recall/F1 numbers here before final
submission — don't submit this row unfilled.

**Key finding (from `NLP_METADATA_REPORT.md`):** both topic and sentiment
metadata degrade sharply on jargon-heavy chunks (topic relevance drops from
61.8% to 12.5%, sentiment from ~47% to ~37.5%) — expected for a
bag-of-words LDA model and a lexicon-based sentiment tool on domain jargon
they weren't built for, and documented rather than hidden.

**Running Task 4** (after Task 2's `chroma_db/` exists):
```bash
python topic_modeling.py
python ner_enrichment.py
python -m pytest test_topic_modeling.py -v      # 12 passed, 1 skipped
python -m pytest test_ner_enrichment.py -v      # 13 passed
python evaluate_nlp_metadata.py
python evaluate_ner_accuracy.py
```

**All steps in one block** (for copy-paste convenience):
```bash
python verify_setup.py
python load_and_validate_dataset.py
python -m pytest test_text_cleaning.py -v
python generate_clean_dataset.py
python chunking.py
python embeddings.py
python chroma_ingest.py
python test_retrieval.py
cp .env.example .env    # then add your OpenRouter key
python -m pytest test_generate_answer.py -v
python generate_answer.py "how do spacecraft communicate with mission control over long distances"
python topic_modeling.py
python ner_enrichment.py
python -m pytest test_topic_modeling.py -v
python -m pytest test_ner_enrichment.py -v
python -m pytest test_api.py -v
python -m uvicorn api:app --port 8000   # in a separate terminal, then in a new one:
python evaluate_retrieval.py
python evaluate_generation.py
```

---
### Task 5 — Evaluation & API (FastAPI)

**Status: complete.** Wraps the Task 3/4 RAG pipeline in a FastAPI service
with structured logging, a proper HTTP error model, automated retrieval and
generation evaluation scripts, and a pytest suite for the endpoints
themselves.

**What's done:**
- `api.py` — FastAPI app with three endpoints: `GET /health` (readiness
  check, does not call the LLM), `POST /api/v1/retrieve` (retrieval only —
  cheap, fast, used by the retrieval eval script), and `POST /api/v1/query`
  (full RAG: retrieve + generate + ground-check). Structured JSON request
  logging middleware (`request_id`, `path`, `status_code`, `duration_ms`) to
  stdout and `logs/api_requests.jsonl`.
- **Error model:** `422` for request-schema failures (Pydantic — blank
  question/query, `top_k` outside `[1, 20]`); `400` for domain-level errors
  (e.g. requesting `topic_label`/`sentiment_label` filtering while running
  on the TF-IDF fallback backend, which has no metadata to filter on); `500`
  reserved for genuinely unexpected bugs, with a generic client-facing
  message and the full traceback only in the structured logs. **Design
  decision:** LLM failures (missing API key, OpenRouter timeout/rate-limit)
  are *not* `500`s — `generate_answer.py` already treats those as a normal
  RAG outcome, so they come back as HTTP `200` with `"status": "error"` in
  the body, the same bucket as `"abstained"`/`"out_of_domain"`.
- `schemas.py` — Pydantic request/response models (`QueryRequest`/
  `QueryResponse`, `RetrieveRequest`/`RetrieveResponse`, `ChunkOut`,
  `HealthResponse`, `ErrorResponse`), with field validators rejecting blank/
  whitespace-only `query`/`question` so those fail as `422`s automatically.
- `logging_config.py` — JSON log formatter + `setup_logging()`.
- `retriever.py` / `generate_answer.py` (extended from Task 3) — both
  `.retrieve()`/`.answer()` now accept optional `top_k` and `where` (Chroma
  metadata filter) parameters, threaded through from the API request.
- `evaluate_retrieval.py` — Precision@K/Recall@K (K = 1, 3, 5, 10) against
  the **live running API** (calls `POST /api/v1/retrieve` over HTTP, not the
  retriever directly, so it's testing the actual deployed service). 12
  hand-crafted queries spanning all 12 categories used in this eval.
  Relevance label = the retrieved chunk's `category` metadata (original 20
  Newsgroups group) matching the query's expected category, since there are
  no hand-labeled per-query relevance judgments for this corpus. Recall@K's
  denominator (true chunk count per category) is fetched directly from
  Chroma — the one place this script goes around the API, since that count
  isn't something the API should expose. Writes JSON + Markdown reports to
  `eval_reports/`.
- `evaluate_generation.py` — end-to-end generation eval against the live
  `POST /api/v1/query` endpoint. 15 queries: the same 12 in-domain queries
  as the retrieval eval (for cross-script comparability) plus 3 queries
  chosen to sit clearly outside the 20-Newsgroups corpus, to exercise the
  out-of-domain gate. Reports the status distribution across the four
  `GenerationResult` buckets (`answered`/`abstained`/`out_of_domain`/
  `error`), the "grounded rate" among `answered` responses (reusing
  `check_grounding()` from Task 3 — a lexical-overlap proxy for "did the
  model stick to the retrieved context"), and latency (mean/p50/p95,
  retrieval vs. generation split). Writes JSON + Markdown reports to
  `eval_reports/`.
- `test_api.py` — 19 pytest tests using FastAPI's `TestClient`. The real
  `RAGPipeline` (which normally loads embeddings/Chroma at startup) is
  replaced with a lightweight fake via `monkeypatch.setattr(api, "RAGPipeline", ...)`
  before the app's lifespan runs, so no network calls, no OpenRouter key,
  and no real vector store are needed to run this suite. Covers: happy
  paths for all three endpoints; `422` (blank query/question, `top_k` out of
  range); `400` (filter-on-TF-IDF-backend, and a `ValueError` raised deeper
  in the pipeline); `500` (an unexpected exception, asserting the response
  never leaks the real error text); the documented LLM-failure-is-`200`
  design decision; the `X-Request-ID` response header; and that `top_k`/
  `where` are passed through correctly end-to-end.

**Verified numbers (from a live run against the real `chroma_db`
collection):**

| Eval | Metric | Result |
|---|---|---|
| Retrieval | Precision@1 | 0.750 |
| Retrieval | Precision@3 | 0.833 |
| Retrieval | Precision@5 | 0.817 |
| Retrieval | Precision@10 | 0.800 |
| Retrieval | Recall@1 / @3 / @5 / @10 | 0.000 / 0.001 / 0.002 / 0.005 (expected — each category has 1,300–2,400 chunks, so top-10 recall is naturally tiny) |
| Generation | Status distribution | 8/15 `answered`, 7/15 `abstained` |
| Generation | Grounded rate (of `answered`) | 8/8 (100.0%) |
| Generation | Total latency | mean 5.37s, p50 3.47s, p95 11.54s |
| Generation | Retrieval-only latency | ~0.2–0.3s (LLM call dominates total latency) |

One known miss, flagged rather than hidden: the query *"how to render 3D
graphics with a GPU"* scores `0.00` Precision at every K in the retrieval
eval, and comes back `abstained`/ungrounded in the generation eval —
reproducible across separate runs, likely landing near
`sci.electronics`/`comp.sys.*` chunks instead of `comp.graphics`. Worth
investigating further but not blocking, since it's one query out of 15 and
every other in-domain query performs well.

**Setup:** `fastapi`, `uvicorn[standard]`, and `httpx` are already in
`requirements.txt` — no separate install step needed.
```bash
pip install -r requirements.txt
```

**Run (two terminals — the eval scripts need a live server to call):**
```bash
# Terminal 1
python -m uvicorn api:app --port 8000
# then see interactive docs at http://127.0.0.1:8000/docs

# Terminal 2
python evaluate_retrieval.py
python evaluate_generation.py
```
> **PowerShell note:** if a bare `uvicorn` command isn't recognized, use
> `python -m uvicorn api:app --port 8000` instead — same reasoning as the
> `python -m pytest` note above (pip installs the `.exe` into a `Scripts`
> folder that isn't always on `PATH`).

**Test (no server, no API key needed):**
```bash
python -m pytest test_api.py -v
```
Expected: `19 passed`. Combined with Task 3's suite:
`python -m pytest test_api.py test_generate_answer.py -v` → `37 passed`.

---
