"""
schemas.py
----------
Pydantic request/response models for api.py.

Validation failures on these models (missing fields, wrong types, blank
strings, out-of-range top_k, etc.) are turned into HTTP 422 responses
automatically by FastAPI -- that's the standard REST convention (422
Unprocessable Entity = "your request was well-formed JSON but the content
is invalid"), and it's why the strip/blank checks live here as field
validators instead of as manual checks inside the endpoint functions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

class ChunkOut(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


def _validate_nonblank(v: str, field_name: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError(f"{field_name} must not be blank")
    return v


# --------------------------------------------------------------------------
# /api/v1/retrieve
# --------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    topic_label: Optional[str] = Field(default=None, max_length=200)
    sentiment_label: Optional[str] = Field(default=None, max_length=50)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        return _validate_nonblank(v, "query")


class RetrieveResponse(BaseModel):
    query: str
    count: int
    backend: str
    chunks: list[ChunkOut]


# --------------------------------------------------------------------------
# /api/v1/query
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    topic_label: Optional[str] = Field(default=None, max_length=200)
    sentiment_label: Optional[str] = Field(default=None, max_length=50)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        return _validate_nonblank(v, "question")


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: str  # "answered" | "abstained" | "out_of_domain" | "error"
    grounded: Optional[bool] = None
    model: str
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    chunks: list[ChunkOut]


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    retriever_backend: str
    dataset_path: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    request_id: Optional[str] = None
