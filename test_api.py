"""
test_api.py
-----------
Unit tests for api.py (Task 5). Uses FastAPI's TestClient. No real pipeline
is ever loaded and no real OpenRouter/network calls are ever made -- the
RAGPipeline that api.py's lifespan normally constructs is replaced with a
lightweight fake for every test, the same way test_generate_answer.py mocks
`requests.post` instead of hitting OpenRouter for real.

Why RAGPipeline is patched at the class level (api.RAGPipeline), not
instance level: api.py's `lifespan()` does
`app.state.pipeline = RAGPipeline(dataset_path=dataset_path)` at startup,
before any test gets a chance to reach into app.state. So the only place to
intercept it is the name `RAGPipeline` as imported into the `api` module.
`monkeypatch.setattr(api, "RAGPipeline", ...)` before `with TestClient(app)`
(which runs the lifespan) does that.

Fake retriever classes are named "ChromaRetriever" / "TfidfRetriever" (via
`__name__` on the stub class, not by subclassing the real ones) because
api.py's backend check is `type(pipeline.retriever).__name__ ==
"TfidfRetriever"` -- it string-compares the class name rather than
isinstance-checking, so the stubs only need to satisfy that.

Coverage:
  - GET / and GET /health (happy paths)
  - POST /api/v1/retrieve: happy path, 422 (blank query, top_k out of
    [1,20]), 400 (topic/sentiment filter requested on the TF-IDF backend)
  - POST /api/v1/query: happy path (status="answered"), 422 (blank
    question), 400 (filter-on-tfidf, and a ValueError raised deeper in the
    pipeline), 500 (an unexpected/unanticipated exception, confirming no
    internal detail leaks into the response body)
  - Cross-cutting: X-Request-ID response header, top_k/where passed through
    correctly to the retriever/pipeline, and that the filter-on-tfidf check
    short-circuits BEFORE the retriever/pipeline is ever called.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api
from generate_answer import GenerationResult
from retriever import RetrievedChunk


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _ChromaRetrieverStub:
    """Fake retriever whose class name is 'ChromaRetriever' (metadata
    filtering allowed)."""

    def __init__(self, chunks=None, side_effect=None):
        self.retrieve = MagicMock(return_value=chunks or [], side_effect=side_effect)


_ChromaRetrieverStub.__name__ = "ChromaRetriever"


class _TfidfRetrieverStub:
    """Fake retriever whose class name is 'TfidfRetriever' (no metadata,
    so api.py should reject topic_label/sentiment_label filters with a 400
    before ever calling .retrieve())."""

    def __init__(self, chunks=None, side_effect=None):
        self.retrieve = MagicMock(return_value=chunks or [], side_effect=side_effect)


_TfidfRetrieverStub.__name__ = "TfidfRetriever"


class _FakePipeline:
    def __init__(self, retriever, answer_return=None, answer_side_effect=None):
        self.retriever = retriever
        self.answer = MagicMock(return_value=answer_return, side_effect=answer_side_effect)


def _client(monkeypatch, fake_pipeline) -> TestClient:
    """Patches api.RAGPipeline so lifespan() hands back our fake instead of
    building a real one, then returns a TestClient. Callers must use this
    as a context manager (`with _client(...) as c:`) so the lifespan
    startup/shutdown actually runs."""
    monkeypatch.setattr(api, "RAGPipeline", lambda dataset_path: fake_pipeline)
    return TestClient(api.app, raise_server_exceptions=False)


def _sample_chunk(doc_id="1", score=0.9, metadata=None) -> RetrievedChunk:
    return RetrievedChunk(text="Apollo 11 landed on the Moon in 1969.", score=score, doc_id=doc_id, metadata=metadata or {})


# --------------------------------------------------------------------------
# GET / and GET /health
# --------------------------------------------------------------------------

def test_root_returns_service_info(monkeypatch):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Parallax Labs RAG API"
    assert "/api/v1/query" in body["endpoints"]


def test_health_reports_retriever_backend(monkeypatch):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["retriever_backend"] == "ChromaRetriever"
    assert "dataset_path" in body


def test_every_response_carries_a_request_id_header(monkeypatch):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.get("/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


# --------------------------------------------------------------------------
# POST /api/v1/retrieve
# --------------------------------------------------------------------------

def test_retrieve_happy_path(monkeypatch):
    chunk = _sample_chunk(doc_id="42", score=0.87, metadata={"topic_label": "sci.space"})
    fake_retriever = _ChromaRetrieverStub(chunks=[chunk])
    fake_pipeline = _FakePipeline(retriever=fake_retriever)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "space travel", "top_k": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "space travel"
    assert body["count"] == 1
    assert body["backend"] == "ChromaRetriever"
    assert body["chunks"][0]["doc_id"] == "42"
    assert body["chunks"][0]["metadata"]["topic_label"] == "sci.space"

    # top_k from the request should be passed straight through
    fake_retriever.retrieve.assert_called_once()
    _, kwargs = fake_retriever.retrieve.call_args
    assert kwargs["top_k"] == 3


def test_retrieve_defaults_top_k_to_4_when_omitted(monkeypatch):
    fake_retriever = _ChromaRetrieverStub(chunks=[])
    fake_pipeline = _FakePipeline(retriever=fake_retriever)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "space travel"})

    assert resp.status_code == 200
    _, kwargs = fake_retriever.retrieve.call_args
    assert kwargs["top_k"] == 4


def test_retrieve_blank_query_is_422(monkeypatch):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "   "})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_type"] == "validation_error"


@pytest.mark.parametrize("bad_top_k", [0, 21, -1])
def test_retrieve_top_k_out_of_range_is_422(monkeypatch, bad_top_k):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "space", "top_k": bad_top_k})
    assert resp.status_code == 422


def test_retrieve_filter_on_tfidf_backend_is_400_and_skips_retriever(monkeypatch):
    fake_retriever = _TfidfRetrieverStub(chunks=[_sample_chunk()])
    fake_pipeline = _FakePipeline(retriever=fake_retriever)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post(
            "/api/v1/retrieve",
            json={"query": "space travel", "topic_label": "sci.space"},
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error_type"] == "bad_request"
    assert "TF-IDF" in body["detail"]
    # the whole point of the check is to reject before doing any retrieval work
    fake_retriever.retrieve.assert_not_called()


def test_retrieve_without_filter_works_fine_on_tfidf_backend(monkeypatch):
    """The TF-IDF restriction only applies when a filter is actually
    requested -- plain queries should work normally on either backend."""
    fake_retriever = _TfidfRetrieverStub(chunks=[_sample_chunk()])
    fake_pipeline = _FakePipeline(retriever=fake_retriever)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/retrieve", json={"query": "space travel"})

    assert resp.status_code == 200
    assert resp.json()["backend"] == "TfidfRetriever"
    fake_retriever.retrieve.assert_called_once()


# --------------------------------------------------------------------------
# POST /api/v1/query
# --------------------------------------------------------------------------

def _answered_result(**overrides) -> GenerationResult:
    defaults = dict(
        answer="The Apollo 11 mission landed astronauts on the Moon in 1969. [1]",
        status="answered",
        grounded=True,
        retrieved_chunks=[_sample_chunk()],
        retrieval_latency_s=0.05,
        generation_latency_s=1.2,
        total_latency_s=1.25,
        model="deepseek/deepseek-chat",
    )
    defaults.update(overrides)
    return GenerationResult(**defaults)


def test_query_happy_path_answered(monkeypatch):
    fake_pipeline = _FakePipeline(
        retriever=_ChromaRetrieverStub(chunks=[_sample_chunk()]),
        answer_return=_answered_result(),
    )

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": "What happened to Apollo 11?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["grounded"] is True
    assert body["model"] == "deepseek/deepseek-chat"
    assert body["chunks"][0]["doc_id"] == "1"
    assert body["total_latency_s"] == pytest.approx(1.25)


def test_query_passes_top_k_and_where_through_to_pipeline(monkeypatch):
    fake_pipeline = _FakePipeline(
        retriever=_ChromaRetrieverStub(),
        answer_return=_answered_result(),
    )

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post(
            "/api/v1/query",
            json={"question": "What happened to Apollo 11?", "top_k": 7, "sentiment_label": "positive"},
        )

    assert resp.status_code == 200
    _, kwargs = fake_pipeline.answer.call_args
    assert kwargs["top_k"] == 7
    assert kwargs["where"] == {"sentiment_label": "positive"}


def test_query_blank_question_is_422(monkeypatch):
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub())
    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": ""})
    assert resp.status_code == 422
    assert resp.json()["error_type"] == "validation_error"


def test_query_filter_on_tfidf_backend_is_400_and_skips_pipeline(monkeypatch):
    fake_pipeline = _FakePipeline(
        retriever=_TfidfRetrieverStub(),
        answer_return=_answered_result(),
    )

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post(
            "/api/v1/query",
            json={"question": "What happened to Apollo 11?", "topic_label": "sci.space"},
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error_type"] == "bad_request"
    assert "TF-IDF" in body["detail"]
    fake_pipeline.answer.assert_not_called()


def test_query_out_of_domain_status_passes_through(monkeypatch):
    out_of_domain_result = GenerationResult(
        answer="I don't have relevant information in the knowledge base to answer this question confidently.",
        status="out_of_domain",
        grounded=None,
        retrieved_chunks=[],
        retrieval_latency_s=0.02,
        generation_latency_s=0.0,
        total_latency_s=0.02,
        model="deepseek/deepseek-chat",
    )
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub(), answer_return=out_of_domain_result)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": "What's a good cookie recipe?"})

    assert resp.status_code == 200  # NOT an HTTP error -- see design note in api.py
    body = resp.json()
    assert body["status"] == "out_of_domain"
    assert body["grounded"] is None


def test_query_valueerror_from_pipeline_is_400(monkeypatch):
    """A ValueError raised deeper in the stack (e.g. the retriever itself
    rejecting something) should map to 400, same bucket as the explicit
    filter-on-tfidf check -- not a 500."""
    fake_pipeline = _FakePipeline(
        retriever=_ChromaRetrieverStub(),
        answer_side_effect=ValueError("unsupported filter combination"),
    )

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": "What happened to Apollo 11?"})

    assert resp.status_code == 400
    body = resp.json()
    assert body["error_type"] == "bad_request"
    assert body["detail"] == "unsupported filter combination"


def test_query_unexpected_exception_is_500_with_generic_message(monkeypatch):
    """LLM failures (missing key, OpenRouter timeout/rate-limit) are NOT
    500s in this design -- generate_answer.py already catches those and
    returns status='error' with HTTP 200 (see test_query_llm_error_status_is_http_200
    below). A 500 here simulates a genuinely unexpected bug, and the test
    asserts the client never sees the real exception text/traceback."""
    fake_pipeline = _FakePipeline(
        retriever=_ChromaRetrieverStub(),
        answer_side_effect=RuntimeError("unexpected null pointer somewhere in pipeline internals"),
    )

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": "What happened to Apollo 11?"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error_type"] == "internal_error"
    assert "unexpected null pointer" not in body["detail"]
    assert body["detail"] == "An internal error occurred. Please try again shortly."


def test_query_llm_error_status_is_http_200_not_500(monkeypatch):
    """Per api.py's documented design decision: OpenRouter/LLM failures are
    a normal (if unfortunate) RAG outcome, not a server error -- they come
    back as HTTP 200 with status='error' in the body, same bucket as
    'abstained'/'out_of_domain'."""
    llm_error_result = GenerationResult(
        answer="Sorry, I couldn't generate an answer right now (API error). Please try again shortly.",
        status="error",
        grounded=None,
        retrieved_chunks=[_sample_chunk()],
        retrieval_latency_s=0.05,
        generation_latency_s=0.01,
        total_latency_s=0.06,
        model="deepseek/deepseek-chat",
        error="Rate limited by OpenRouter",
    )
    fake_pipeline = _FakePipeline(retriever=_ChromaRetrieverStub(), answer_return=llm_error_result)

    with _client(monkeypatch, fake_pipeline) as client:
        resp = client.post("/api/v1/query", json={"question": "What happened to Apollo 11?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
