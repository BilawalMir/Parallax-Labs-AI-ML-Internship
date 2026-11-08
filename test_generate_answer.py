"""
test_generate_answer.py
------------------------
Unit tests for generate_answer.py. The OpenRouter API is mocked throughout
(no network calls, no API key needed to run these tests) so this test suite
runs the same in CI as it does locally.
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

import generate_answer as ga
from retriever import RetrievedChunk


# --------------------------------------------------------------------------
# Prompt engineering
# --------------------------------------------------------------------------

def test_build_context_block_numbers_chunks_in_score_order():
    chunks = [
        RetrievedChunk(text="low score chunk", score=0.1, doc_id=0),
        RetrievedChunk(text="high score chunk", score=0.9, doc_id=1),
    ]
    block = ga.build_context_block(chunks)
    assert block.startswith("[1] high score chunk")
    assert "[2] low score chunk" in block


def test_build_context_block_respects_token_budget():
    long_text = "word " * 5000
    chunks = [RetrievedChunk(text=long_text, score=0.9, doc_id=0),
              RetrievedChunk(text="short chunk", score=0.5, doc_id=1)]
    block = ga.build_context_block(chunks, max_tokens=50)
    # only the first (highest scoring) chunk should be kept when it alone
    # already exceeds the budget
    assert "[2]" not in block


def test_build_messages_includes_system_prompt_and_question():
    messages = ga.build_messages("What is X?", "[1] some context")
    assert messages[0]["role"] == "system"
    assert "only" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "What is X?" in messages[1]["content"]
    assert "[1] some context" in messages[1]["content"]


# --------------------------------------------------------------------------
# Hallucination / grounding check
# --------------------------------------------------------------------------

def test_check_grounding_high_overlap_passes():
    context = "The Apollo 11 mission landed astronauts Armstrong and Aldrin on the Moon in 1969."
    answer = "Armstrong and Aldrin landed on the Moon during the Apollo 11 mission."
    assert ga.check_grounding(answer, context) is True


def test_check_grounding_low_overlap_fails():
    context = "The Apollo 11 mission landed astronauts on the Moon in 1969."
    answer = "Quantum entanglement enables superluminal cryptographic teleportation networks."
    assert ga.check_grounding(answer, context) is False


def test_check_grounding_empty_answer_is_trivially_grounded():
    assert ga.check_grounding("", "some context") is True


# --------------------------------------------------------------------------
# API error handling (mocked)
# --------------------------------------------------------------------------

def _mock_response(status_code=200, json_body=None, text="", headers=None):
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("no json")
    return resp


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_call_openrouter_success(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "  The answer is X. [1]  "}}]}
    )
    result = ga.call_openrouter([{"role": "user", "content": "hi"}])
    assert result == "The answer is X. [1]"


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_call_openrouter_missing_key_raises_before_network_call(mock_post):
    with patch.object(ga, "OPENROUTER_API_KEY", None):
        with pytest.raises(ga.OpenRouterError, match="OPENROUTER_API_KEY"):
            ga.call_openrouter([{"role": "user", "content": "hi"}])
    mock_post.assert_not_called()


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.time.sleep", return_value=None)  # skip real backoff delays
@patch("generate_answer.requests.post")
def test_call_openrouter_retries_on_429_then_succeeds(mock_post, mock_sleep):
    mock_post.side_effect = [
        _mock_response(429, headers={"Retry-After": "1"}),
        _mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
    ]
    result = ga.call_openrouter([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert mock_post.call_count == 2


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.time.sleep", return_value=None)
@patch("generate_answer.requests.post")
def test_call_openrouter_exhausts_retries_on_persistent_429(mock_post, mock_sleep):
    mock_post.return_value = _mock_response(429, headers={"Retry-After": "1"})
    with pytest.raises(ga.OpenRouterError, match="Rate limited"):
        ga.call_openrouter([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == ga.MAX_RETRIES


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_call_openrouter_context_length_error_is_not_retried(mock_post):
    mock_post.return_value = _mock_response(
        400, {"error": {"message": "maximum context length exceeded"}}
    )
    with pytest.raises(ga.OpenRouterError, match="context window"):
        ga.call_openrouter([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 1  # not retried -- it's a client error


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.time.sleep", return_value=None)
@patch("generate_answer.requests.post")
def test_call_openrouter_timeout_is_retried_then_raises(mock_post, mock_sleep):
    mock_post.side_effect = requests.exceptions.Timeout()
    with pytest.raises(ga.OpenRouterError, match="Failed after"):
        ga.call_openrouter([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == ga.MAX_RETRIES


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_call_openrouter_malformed_response_raises(mock_post):
    mock_post.return_value = _mock_response(200, {"unexpected": "shape"})
    with pytest.raises(ga.OpenRouterError, match="Malformed"):
        ga.call_openrouter([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------
# Full pipeline (mocked API)
# --------------------------------------------------------------------------

@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import pandas as pd

    csv_path = tmp_path / "processed_dataset.csv"
    pd.DataFrame({"text": [
        "The Space Shuttle Columbia disaster occurred in 2003 during re-entry.",
        "Apollo 11 landed the first humans on the Moon in 1969.",
    ]}).to_csv(csv_path, index=False)

    monkeypatch.chdir(tmp_path)
    return ga.RAGPipeline(dataset_path=str(csv_path))


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_pipeline_answers_in_domain_query(mock_post, pipeline):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "The Columbia disaster happened in 2003. [1]"}}]}
    )
    result = pipeline.answer("What happened to the Columbia shuttle?")
    assert result.status == "answered"
    assert "2003" in result.answer
    assert result.retrieval_latency_s >= 0
    assert result.generation_latency_s >= 0
    assert result.total_latency_s >= result.retrieval_latency_s


def test_pipeline_flags_out_of_domain_query_without_calling_api(pipeline):
    with patch("generate_answer.requests.post") as mock_post:
        result = pipeline.answer("What is the capital of a fictional planet Zorblax-9?")
        assert result.status == "out_of_domain"
        mock_post.assert_not_called()  # out-of-domain queries should never hit the paid API


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_pipeline_marks_abstained_when_model_declines(mock_post, pipeline):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "I cannot answer this from the provided context."}}]}
    )
    result = pipeline.answer("What happened to the Columbia shuttle?")
    assert result.status == "abstained"


@patch.object(ga, "OPENROUTER_API_KEY", "fake-key-for-tests")
@patch("generate_answer.requests.post")
def test_pipeline_handles_api_error_gracefully(mock_post, pipeline):
    mock_post.return_value = _mock_response(500)
    with patch("generate_answer.time.sleep", return_value=None):
        result = pipeline.answer("What happened to the Columbia shuttle?")
    assert result.status == "error"
    assert result.error is not None
    assert "couldn't generate" in result.answer.lower()


def test_pipeline_logs_every_query(pipeline, tmp_path):
    with patch("generate_answer.requests.post") as mock_post:
        pipeline.answer("What is the capital of a fictional planet Zorblax-9?")
    log_path = tmp_path / ga.LOG_FILE
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "out_of_domain"
    assert "total_latency_s" in record
