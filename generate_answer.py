"""
generate_answer.py
-------------------
Task: integrate DeepSeek (via OpenRouter) to generate answers from retrieved
chunks, on top of the Week 1/2 pipeline.

Covers all four deliverable requirements:
  1. Prompt engineering  -> build_messages() (system prompt + context injection)
  2. Robust API error handling -> call_openrouter() (timeouts, rate limits,
     token limits, malformed responses, retries w/ exponential backoff)
  3. Hallucination / out-of-domain handling -> retrieval-confidence gate +
     grounding check on the generated answer (check_grounding())
  4. Latency logging -> RAGPipeline.answer() times retrieval + generation
     separately and logs every query to rag_latency_log.jsonl

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...      # see README / .env.example
    python generate_answer.py "What caused the Columbia disaster?"

    # or interactively:
    python generate_answer.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv

from retriever import RetrievedChunk, get_retriever

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

load_dotenv()  # reads .env if present, never overrides real env vars set by the shell

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# "deepseek/deepseek-chat" is OpenRouter's stable alias for DeepSeek's flagship
# chat model (currently routes to DeepSeek V3.2) -- good default: solid quality,
# cheap, always points at DeepSeek's current non-reasoning flagship instead of a
# dated snapshot. For $0 experimentation while building/testing, swap in
# "deepseek/deepseek-chat-v3.1:free" via OPENROUTER_MODEL (free tier is rate
# limited and not meant for production use). Full current list, pricing, and
# free variants: https://openrouter.ai/deepseek
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

TOP_K = 4
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.08  # below this, treat query as out-of-domain
MAX_CONTEXT_TOKENS = 2000              # budget for the injected context block
MAX_OUTPUT_TOKENS = 500
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2

LOG_FILE = "rag_latency_log.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag")

# Token counting: tiktoken doesn't ship an official DeepSeek encoding, so we
# use cl100k_base as a reasonable proxy for budgeting context length. It
# won't be exact but it's good enough to avoid blowing the context window.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens in `text` using the cl100k_base encoding (proxy for
        DeepSeek's actual tokenizer, since tiktoken has no official one)."""
        return len(_ENC.encode(text))

except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    def count_tokens(text: str) -> int:
        """Crude fallback token estimate (~4 chars/token) used only if
        tiktoken isn't installed."""
        return max(1, len(text) // 4)  # crude ~4 chars/token estimate


SYSTEM_PROMPT = """You are a careful, factual assistant that answers questions using ONLY the \
numbered context excerpts provided below. Follow these rules strictly:

1. Base your answer only on the provided context. Do not use outside knowledge \
and do not guess.
2. If the context does not contain enough information to answer the question, \
respond with exactly: "I cannot answer this from the provided context." \
Do not attempt a partial or speculative answer in that case.
3. When you use information from a chunk, cite it inline like [1], [2], etc., \
matching the chunk numbers in the context.
4. Be concise. Do not pad the answer with information not asked for.
5. Never fabricate sources, statistics, quotes, or citations that are not in \
the context.
"""

OUT_OF_DOMAIN_MESSAGE = (
    "I don't have relevant information in the knowledge base to answer this "
    "question confidently, so I won't guess. Try rephrasing, or ask something "
    "closer to the dataset's topic area."
)

ABSTAIN_PHRASE = "i cannot answer this from the provided context"


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Everything one call to `RAGPipeline.answer()` produces.

    `status` tells you how to interpret `answer`: "answered" is a normal
    grounded response, "abstained" means the model itself declined (per
    the system prompt's abstain rule), "out_of_domain" means retrieval
    never reached the LLM because no chunk cleared
    `RETRIEVAL_CONFIDENCE_THRESHOLD`, and "error" means the OpenRouter
    call failed after retries (see `error` for details).
    """

    answer: str
    status: str  # "answered" | "abstained" | "out_of_domain" | "error"
    grounded: Optional[bool] = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    model: str = MODEL_NAME
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Prompt engineering: context injection
# --------------------------------------------------------------------------

def build_context_block(chunks: list[RetrievedChunk], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Format retrieved chunks into a numbered context block, trimming the
    lowest-scoring chunks first if the block would exceed the token budget."""
    chunks_sorted = sorted(chunks, key=lambda c: c.score, reverse=True)
    kept: list[str] = []
    running_tokens = 0
    for i, chunk in enumerate(chunks_sorted, start=1):
        piece = f"[{i}] {chunk.text.strip()}"
        piece_tokens = count_tokens(piece)
        if running_tokens + piece_tokens > max_tokens and kept:
            # keep at least one chunk even if it's over budget alone
            break
        kept.append(piece)
        running_tokens += piece_tokens
    return "\n\n".join(kept)


def build_messages(question: str, context_block: str) -> list[dict]:
    """Assemble the OpenRouter chat `messages` payload for one question.

    Wraps `question` and `context_block` into a user message instructing
    the model to answer only from the given context with inline citations,
    paired with the module-level `SYSTEM_PROMPT`.

    Args:
        question: The user's question.
        context_block: Numbered context text from `build_context_block()`.

    Returns:
        A `[{"role": "system", ...}, {"role": "user", ...}]` list ready to
        pass to `call_openrouter()`.
    """
    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations like [1]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------
# Robust OpenRouter API call
# --------------------------------------------------------------------------

class OpenRouterError(Exception):
    """Raised for API failures after retries are exhausted."""


def call_openrouter(messages: list[dict], max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """Call the OpenRouter chat completions API with retries and backoff.

    Handles the failure modes an internship-grade RAG service needs to
    survive: request timeouts and connection errors (retried), 429 rate
    limiting (retried, honoring `Retry-After` if present), 5xx server
    errors (retried with exponential backoff), 4xx client errors
    (not retried -- e.g. bad key, prompt too long for the context window),
    and malformed/unexpected response bodies.

    Args:
        messages: Chat messages in OpenRouter/OpenAI format, typically from
            `build_messages()`.
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        The generated answer text, stripped of leading/trailing whitespace.

    Raises:
        OpenRouterError: If `OPENROUTER_API_KEY` is unset, a non-retryable
            client error occurs, the response body is malformed, or all
            `MAX_RETRIES` attempts are exhausted.
    """
    if not OPENROUTER_API_KEY:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://openrouter.ai/keys, or `export "
            "OPENROUTER_API_KEY=sk-or-...` before running."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter asks for these for attribution/rate-limit purposes; optional but polite.
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/Maryam-Yaqoob/Parallax-Labs-AI-ML-Internship"),
        "X-Title": "Parallax Labs RAG Internship Task",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,  # low temperature: factual RAG answers, not creative writing
    }

    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s"
            logger.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, last_error)
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, last_error)
        else:
            # --- rate limiting ---
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                last_error = "Rate limited (429)"
                logger.warning(
                    "Attempt %d/%d: rate limited, backing off %.1fs", attempt, MAX_RETRIES, wait
                )
                time.sleep(wait)
                continue

            # --- server errors: retryable ---
            if response.status_code >= 500:
                last_error = f"Server error {response.status_code}"
                logger.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, last_error)
                time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                continue

            # --- client errors: not retryable (bad key, bad request, context length, etc.) ---
            if response.status_code >= 400:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    detail = response.text
                # Context-length / token-limit errors surface as 400s from OpenRouter.
                if response.status_code == 400 and "context" in detail.lower():
                    raise OpenRouterError(
                        f"Prompt too long for model context window: {detail}"
                    )
                raise OpenRouterError(f"API error {response.status_code}: {detail}")

            # --- success path ---
            try:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except (ValueError, KeyError, IndexError) as e:
                raise OpenRouterError(f"Malformed API response: {e} | raw={response.text[:300]}")

        # only reached after a timeout/connection error above
        time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    raise OpenRouterError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


# --------------------------------------------------------------------------
# Hallucination check
# --------------------------------------------------------------------------

def check_grounding(answer: str, context_block: str, min_overlap: float = 0.25) -> bool:
    """
    Lightweight lexical-grounding heuristic: what fraction of the answer's
    "content" words (4+ letters, deduped) also appear somewhere in the
    context block? Low overlap is a signal the model may have drifted from
    the provided context. This is intentionally cheap (no extra API call) --
    it's a guardrail, not a proof of factuality.
    """
    def content_words(text: str) -> set[str]:
        """Extract lowercased, deduped 4+ letter words from `text`."""
        return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())}

    answer_words = content_words(answer)
    if not answer_words:
        return True  # nothing to check (e.g. abstain message)

    context_words = content_words(context_block)
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end retrieve-then-generate pipeline used by the CLI and api.py."""

    def __init__(self, dataset_path: str = "processed_dataset.csv"):
        """Build the pipeline's retriever via `get_retriever()`.

        Args:
            dataset_path: Path to the processed CSV, used only by the
                TF-IDF fallback if no Chroma store is found.
        """
        self.retriever = get_retriever(dataset_path)

    def answer(self, question: str, top_k: int | None = None, where: dict | None = None) -> GenerationResult:
        """Run one question through retrieval, the out-of-domain gate, and
        generation, logging latency and outcome along the way.

        Flow: retrieve top-k chunks -> if none clear the confidence
        threshold, return an "out_of_domain" result without calling the
        LLM -> otherwise build the context block and call OpenRouter,
        catching `OpenRouterError` into an "error" result -> classify the
        raw answer as "answered" or "abstained" and run the grounding
        check -> log and return.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve; defaults to `TOP_K`.
            where: Optional Chroma metadata filter, passed through to the
                retriever (ignored/rejected by the TF-IDF backend).

        Returns:
            A `GenerationResult` with `status` one of "answered",
            "abstained", "out_of_domain", or "error", plus latency
            breakdown and the retrieved chunks.
        """
        t_start = time.perf_counter()

        # ---- retrieval ----
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k or TOP_K, where=where)
        retrieval_latency = time.perf_counter() - t0

        # ---- out-of-domain gate ----
        top_score = chunks[0].score if chunks else 0.0
        if not chunks or top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
            total_latency = time.perf_counter() - t_start
            result = GenerationResult(
                answer=OUT_OF_DOMAIN_MESSAGE,
                status="out_of_domain",
                retrieved_chunks=chunks,
                retrieval_latency_s=retrieval_latency,
                generation_latency_s=0.0,
                total_latency_s=total_latency,
            )
            self._log(question, result)
            return result

        context_block = build_context_block(chunks)
        messages = build_messages(question, context_block)

        # ---- generation ----
        t1 = time.perf_counter()
        try:
            raw_answer = call_openrouter(messages)
            status = "answered"
        except OpenRouterError as e:
            generation_latency = time.perf_counter() - t1
            total_latency = time.perf_counter() - t_start
            logger.error("Generation failed: %s", e)
            result = GenerationResult(
                answer="Sorry, I couldn't generate an answer right now (API error). Please try again shortly.",
                status="error",
                retrieved_chunks=chunks,
                retrieval_latency_s=retrieval_latency,
                generation_latency_s=generation_latency,
                total_latency_s=total_latency,
                error=str(e),
            )
            self._log(question, result)
            return result
        generation_latency = time.perf_counter() - t1

        if raw_answer.strip().lower().startswith(ABSTAIN_PHRASE):
            status = "abstained"

        grounded = check_grounding(raw_answer, context_block)
        if status == "answered" and not grounded:
            logger.warning("Low lexical grounding detected for query: %r", question)

        total_latency = time.perf_counter() - t_start
        result = GenerationResult(
            answer=raw_answer,
            status=status,
            grounded=grounded,
            retrieved_chunks=chunks,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
        )
        self._log(question, result)
        return result

    @staticmethod
    def _log(question: str, result: GenerationResult) -> None:
        """Append one JSON record for this query to `LOG_FILE` and emit an
        INFO-level summary line via `logger`.

        Args:
            question: The user's question.
            result: The `GenerationResult` produced for this question.
        """
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": question,
            "status": result.status,
            "grounded": result.grounded,
            "model": result.model,
            "retrieval_latency_s": round(result.retrieval_latency_s, 4),
            "generation_latency_s": round(result.generation_latency_s, 4),
            "total_latency_s": round(result.total_latency_s, 4),
            "num_chunks_retrieved": len(result.retrieved_chunks),
            "top_retrieval_score": round(result.retrieved_chunks[0].score, 4) if result.retrieved_chunks else None,
            "error": result.error,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "status=%s | retrieval=%.3fs | generation=%.3fs | total=%.3fs",
            result.status,
            result.retrieval_latency_s,
            result.generation_latency_s,
            result.total_latency_s,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    """CLI entry point.

    Loads `processed_dataset.csv` (falling back to the bundled sample
    dataset if it's missing) and either answers a single question passed
    as command-line arguments, or drops into an interactive prompt loop
    that answers questions until the user types "exit"/"quit" or hits
    Ctrl-C/EOF.
    """
    dataset_path = "processed_dataset.csv"
    if not os.path.exists(dataset_path):
        fallback = "sample_processed_dataset.csv"
        print(
            f"[info] '{dataset_path}' not found (run generate_clean_dataset.py from "
            f"Task 1 to produce the real dataset). Falling back to the small demo "
            f"dataset '{fallback}' so this script still runs end-to-end.\n"
        )
        dataset_path = fallback
    pipeline = RAGPipeline(dataset_path=dataset_path)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = pipeline.answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        print(f"\n[status={result.status} | total_latency={result.total_latency_s:.2f}s]")
        return

    print("RAG CLI — type a question (or 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = pipeline.answer(question)
        print(f"A: {result.answer}")
        print(f"[status={result.status} | total_latency={result.total_latency_s:.2f}s]")


if __name__ == "__main__":
    main()
