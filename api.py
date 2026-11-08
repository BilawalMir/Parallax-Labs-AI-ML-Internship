"""
api.py
------
FastAPI service wrapping the RAG pipeline (retriever.py + generate_answer.py,
Task 3) with topic/sentiment metadata filtering (Task 4).

Endpoints
---------
GET  /                     service info
GET  /health                readiness check (does NOT call the LLM)
POST /api/v1/retrieve       retrieval only, no generation -- cheap, fast,
                             useful for the retrieval-evaluation script
POST /api/v1/query          full RAG: retrieve + generate + ground-check

Error model
-----------
- 422 Unprocessable Entity: request body failed schema validation (Pydantic) --
  e.g. blank question, top_k out of [1, 20], wrong types. Handled automatically
  by FastAPI/Pydantic; we just reformat the response body for consistency.
- 400 Bad Request: well-formed request that's invalid at the domain level --
  e.g. topic_label/sentiment_label filters requested while running on the
  TF-IDF fallback backend, which has no metadata to filter on.
- 500 Internal Server Error: anything unexpected (bugs, a crashed retriever,
  etc.). The client gets a generic message; the real exception + traceback
  goes to the structured logs, never to the response body.

Note on LLM failures specifically: OPENROUTER_API_KEY missing, OpenRouter
timeouts/5xx/rate-limits, etc. are NOT surfaced as HTTP 500. generate_answer.py
already treats those as a normal (if unfortunate) RAG outcome -- RAGPipeline
.answer() catches OpenRouterError internally and returns a GenerationResult
with status="error" and a friendly message. This endpoint preserves that
design: those responses come back as HTTP 200 with "status": "error" in the
body, exactly like "status": "abstained" or "out_of_domain" do. HTTP 5xx is
reserved for failures the pipeline itself didn't anticipate.

Run
---
    uvicorn api:app --reload --port 8000
    # then see interactive docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from generate_answer import RAGPipeline
from logging_config import new_request_id, setup_logging
from schemas import (
    ChunkOut,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)

logger = setup_logging()


class BadRequestError(Exception):
    """Domain-level 400s: valid JSON/schema, invalid given current pipeline state."""


# --------------------------------------------------------------------------
# Startup: load the RAG pipeline once, not per-request
# --------------------------------------------------------------------------

def _resolve_dataset_path() -> str:
    """Same fallback generate_answer.py's CLI uses: prefer the real cleaned
    dataset, fall back to the small demo dataset so the service still boots."""
    primary = os.getenv("DATASET_PATH", "processed_dataset.csv")
    if os.path.exists(primary):
        return primary
    fallback = "sample_processed_dataset.csv"
    logger.warning(
        "dataset_fallback",
        extra={"event": "dataset_fallback", "requested": primary, "using": fallback},
    )
    return fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: build the RAG pipeline once at startup and
    tear down cleanly at shutdown.

    Loading `RAGPipeline` here (instead of per-request) is what makes the
    service fast -- the embedding model and Chroma/TF-IDF index are loaded
    once and reused for every request via `app.state.pipeline`.

    Args:
        app: The FastAPI application instance being started.

    Yields:
        Control back to FastAPI while the app serves requests; runs the
        shutdown log line after the app stops.
    """
    dataset_path = _resolve_dataset_path()
    t0 = time.perf_counter()
    app.state.pipeline = RAGPipeline(dataset_path=dataset_path)
    app.state.dataset_path = dataset_path
    app.state.retriever_backend = type(app.state.pipeline.retriever).__name__
    logger.info(
        "startup_complete",
        extra={
            "event": "startup_complete",
            "dataset_path": dataset_path,
            "retriever_backend": app.state.retriever_backend,
            "startup_s": round(time.perf_counter() - t0, 3),
        },
    )
    yield
    logger.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(
    title="Parallax Labs RAG API",
    description="FastAPI wrapper around the 20-Newsgroups RAG pipeline (retrieval + DeepSeek generation).",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive default for local/dev use. Tighten allow_origins before any
# real deployment -- this is flagged rather than silently left wide open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Structured request logging middleware
# --------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs a start/finish (or failure) line for every request, with a
    shared `request_id` used to correlate them and returned to the client
    via the `X-Request-ID` response header."""

    async def dispatch(self, request: Request, call_next):
        """Assign a request ID, log request start/finish/failure, and time
        the request.

        Args:
            request: The incoming Starlette/FastAPI request.
            call_next: Callable that runs the rest of the middleware chain
                and the route handler, returning the response.

        Returns:
            The response from `call_next`, with `X-Request-ID` added.

        Raises:
            Exception: Re-raises any exception from `call_next` after
                logging it, so the registered exception handlers still run.
        """
        request_id = new_request_id()
        request.state.request_id = request_id
        t0 = time.perf_counter()

        logger.info(
            "request_started",
            extra={
                "event": "request_started",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.exception(
                "request_unhandled_exception",
                extra={
                    "event": "request_failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_finished",
            extra={
                "event": "request_finished",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# --------------------------------------------------------------------------
# Exception handlers -> consistent ErrorResponse body on every error path
# --------------------------------------------------------------------------

def _request_id_of(request: Request) -> Optional[str]:
    """Fetch the request ID that `RequestLoggingMiddleware` stashed on
    `request.state`, or `None` if the middleware didn't run (e.g. very
    early startup failures)."""
    return getattr(request.state, "request_id", None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Turn Pydantic schema-validation failures into a 422 `ErrorResponse`.

    Args:
        request: The request that failed validation.
        exc: The validation error, containing per-field error details.

    Returns:
        A JSON 422 response with a flattened `field: message` detail string.
    """
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    logger.warning(
        "validation_error",
        extra={"event": "validation_error", "request_id": _request_id_of(request), "detail": detail},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail=detail, error_type="validation_error", request_id=_request_id_of(request)
        ).model_dump(),
    )


@app.exception_handler(BadRequestError)
async def bad_request_handler(request: Request, exc: BadRequestError):
    """Turn a domain-level `BadRequestError` (e.g. metadata filter requested
    on the TF-IDF backend) into a 400 `ErrorResponse`.

    Args:
        request: The request that triggered the error.
        exc: The raised `BadRequestError`, whose message becomes `detail`.

    Returns:
        A JSON 400 response.
    """
    logger.warning(
        "bad_request",
        extra={"event": "bad_request", "request_id": _request_id_of(request), "detail": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            detail=str(exc), error_type="bad_request", request_id=_request_id_of(request)
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Turn a bare `ValueError` from the pipeline/retriever layer into a
    400 `ErrorResponse`, treating it as a domain-level bad request rather
    than an unexpected 500.

    Args:
        request: The request that triggered the error.
        exc: The raised `ValueError` (e.g. `TfidfRetriever` rejecting a
            metadata filter it can't honor).

    Returns:
        A JSON 400 response.
    """
    # Retriever-layer ValueErrors (e.g. TfidfRetriever rejecting a metadata
    # filter it can't honor) are domain-level bad requests, same as above.
    logger.warning(
        "value_error",
        extra={"event": "value_error", "request_id": _request_id_of(request), "detail": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            detail=str(exc), error_type="bad_request", request_id=_request_id_of(request)
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Reformat Starlette's default HTTPException body into the same
    `ErrorResponse` shape as every other error path (e.g. for a 404 on an
    unknown route).

    Args:
        request: The request that triggered the exception.
        exc: The Starlette HTTP exception, carrying its own status code.

    Returns:
        A JSON response using `exc.status_code`.
    """
    logger.warning(
        "http_exception",
        extra={
            "event": "http_exception",
            "request_id": _request_id_of(request),
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail), error_type="http_error", request_id=_request_id_of(request)
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for anything not handled above (bugs, a crashed retriever,
    etc.), returning a generic 500 `ErrorResponse` so internals are never
    leaked to the client. The real exception and traceback are logged by
    `RequestLoggingMiddleware`, not returned here.

    Args:
        request: The request that triggered the unhandled exception.
        exc: The unhandled exception.

    Returns:
        A JSON 500 response with a generic, non-identifying message.
    """
    # Full traceback goes to structured logs (see logger.exception in the
    # middleware above); the client only ever gets a generic message so we
    # never leak internals (API keys, file paths, stack traces) in a response.
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An internal error occurred. Please try again shortly.",
            error_type="internal_error",
            request_id=_request_id_of(request),
        ).model_dump(),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _build_where(topic_label: Optional[str], sentiment_label: Optional[str]) -> Optional[dict]:
    """Build a Chroma `where` metadata filter from the optional request
    filter fields.

    Args:
        topic_label: Topic label to filter on, or `None`.
        sentiment_label: Sentiment label to filter on, or `None`.

    Returns:
        `None` if neither filter is given; a single-clause dict if only
        one is given; otherwise a `{"$and": [...]}` combining both.
    """
    clauses = []
    if topic_label:
        clauses.append({"topic_label": topic_label})
    if sentiment_label:
        clauses.append({"sentiment_label": sentiment_label})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _to_chunk_out(chunk) -> ChunkOut:
    """Convert a `RetrievedChunk` (from retriever.py) into the `ChunkOut`
    API response model.

    Args:
        chunk: A `RetrievedChunk` returned by either retriever backend.

    Returns:
        The equivalent `ChunkOut`, coercing `doc_id` to `str` and
        defaulting `metadata` to `{}` if it's falsy.
    """
    return ChunkOut(doc_id=str(chunk.doc_id), text=chunk.text, score=chunk.score, metadata=chunk.metadata or {})


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    """Service info: name and links to the docs, health check, and API
    endpoints. Useful as a quick sanity check that the service is up."""
    return {
        "service": "Parallax Labs RAG API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/api/v1/retrieve", "/api/v1/query"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request):
    """Readiness check. Reports which retriever backend is active and which
    dataset was loaded, without calling the LLM or touching the retriever.

    Args:
        request: Used only to read `app.state` (set once at startup by
            `lifespan()`).

    Returns:
        A `HealthResponse` with status "ok".
    """
    return HealthResponse(
        status="ok",
        retriever_backend=request.app.state.retriever_backend,
        dataset_path=request.app.state.dataset_path,
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse, tags=["rag"])
async def retrieve(payload: RetrieveRequest, request: Request):
    """Retrieval-only endpoint: run the retriever and return matching
    chunks, with no LLM call. Cheap and fast -- also what
    `evaluate_retrieval.py` hits for the retrieval benchmark.

    Args:
        payload: Validated `RetrieveRequest` (query, top_k, optional
            topic_label/sentiment_label filters).
        request: Used to reach the shared `RAGPipeline` on `app.state`.

    Returns:
        A `RetrieveResponse` with the matched chunks and which backend
        served them.

    Raises:
        BadRequestError: If a metadata filter is requested while running
            on the TF-IDF fallback backend, which has no metadata.
    """
    pipeline: RAGPipeline = request.app.state.pipeline
    where = _build_where(payload.topic_label, payload.sentiment_label)

    if where and type(pipeline.retriever).__name__ == "TfidfRetriever":
        raise BadRequestError(
            "topic_label/sentiment_label filtering requires the Chroma retriever "
            "backend, but this service is currently running on the TF-IDF fallback "
            "(no chroma_db/ found). Run chroma_ingest.py, topic_modeling.py, and "
            "sentiment_analysis.py, then restart the service."
        )

    chunks = pipeline.retriever.retrieve(payload.query, top_k=payload.top_k or 4, where=where)
    return RetrieveResponse(
        query=payload.query,
        count=len(chunks),
        backend=type(pipeline.retriever).__name__,
        chunks=[_to_chunk_out(c) for c in chunks],
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["rag"])
async def query(payload: QueryRequest, request: Request):
    """Full RAG endpoint: retrieve chunks, generate a grounded answer via
    `RAGPipeline.answer()`, and return it. LLM-side failures (missing key,
    OpenRouter timeouts/5xx/rate-limits) come back as HTTP 200 with
    `status="error"` in the body, not as HTTP 500 -- see the module
    docstring's "Error model" section for why.

    Args:
        payload: Validated `QueryRequest` (question, top_k, optional
            topic_label/sentiment_label filters).
        request: Used to reach the shared `RAGPipeline` on `app.state`.

    Returns:
        A `QueryResponse` with the answer, its status, grounding flag,
        latency breakdown, and the chunks used.

    Raises:
        BadRequestError: If a metadata filter is requested while running
            on the TF-IDF fallback backend, which has no metadata.
    """
    pipeline: RAGPipeline = request.app.state.pipeline
    where = _build_where(payload.topic_label, payload.sentiment_label)

    if where and type(pipeline.retriever).__name__ == "TfidfRetriever":
        raise BadRequestError(
            "topic_label/sentiment_label filtering requires the Chroma retriever "
            "backend, but this service is currently running on the TF-IDF fallback "
            "(no chroma_db/ found). Run chroma_ingest.py, topic_modeling.py, and "
            "sentiment_analysis.py, then restart the service."
        )

    result = pipeline.answer(payload.question, top_k=payload.top_k, where=where)
    return QueryResponse(
        question=payload.question,
        answer=result.answer,
        status=result.status,
        grounded=result.grounded,
        model=result.model,
        retrieval_latency_s=result.retrieval_latency_s,
        generation_latency_s=result.generation_latency_s,
        total_latency_s=result.total_latency_s,
        chunks=[_to_chunk_out(c) for c in result.retrieved_chunks],
    )
