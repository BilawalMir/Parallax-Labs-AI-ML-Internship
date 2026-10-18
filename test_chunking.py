"""
test_chunking.py
-----------------
Unit tests for chunking.py — the recursive character chunking strategy
used to prepare processed_dataset.csv (Task 1 output) for embedding.

Run: python -m pytest test_chunking.py -v
"""

import pandas as pd
import pytest

from chunking import (
    recursive_character_split,
    chunk_document,
    chunk_dataframe,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------
# recursive_character_split — edge cases
# ---------------------------------------------------------------------


def test_none_input_returns_empty_list():
    assert recursive_character_split(None) == []


def test_empty_string_returns_empty_list():
    assert recursive_character_split("") == []


def test_whitespace_only_returns_empty_list():
    assert recursive_character_split("   \n\t  ") == []


def test_non_string_input_returns_empty_list():
    assert recursive_character_split(12345) == []
    assert recursive_character_split(["not", "a", "string"]) == []


def test_short_text_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = recursive_character_split(text, chunk_size=1000, chunk_overlap=100)
    assert chunks == [text]


def test_text_exactly_chunk_size_returns_single_chunk():
    text = "a" * 1000
    chunks = recursive_character_split(text, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap_greater_than_or_equal_to_chunk_size_is_neutralized():
    # Should not raise / infinite loop; overlap gets clamped to 0 internally.
    text = "word " * 500  # 2500 chars
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


# ---------------------------------------------------------------------
# recursive_character_split — core splitting behavior
# ---------------------------------------------------------------------


def test_long_text_splits_into_multiple_chunks():
    paragraph = "Sentence one. Sentence two. Sentence three. " * 40  # ~1880 chars
    chunks = recursive_character_split(paragraph, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # No chunk should wildly exceed chunk_size.
    assert all(len(c) <= 500 for c in chunks)


def test_chunks_respect_paragraph_boundaries_when_possible():
    text = (
        "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three is here."
    )
    chunks = recursive_character_split(text, chunk_size=30, chunk_overlap=5)
    # Every paragraph's core sentence should still be findable across chunks.
    joined = "".join(chunks)
    assert "Paragraph one" in joined
    assert "Paragraph two" in joined
    assert "Paragraph three" in joined


def test_no_content_is_lost_without_overlap():
    # With overlap=0, concatenating chunks should reproduce the original text.
    text = ("word " * 300).strip()
    chunks = recursive_character_split(text, chunk_size=200, chunk_overlap=0)
    assert "".join(chunks).strip() == text or "".join(chunks) == text


def test_overlap_produces_shared_context_between_consecutive_chunks():
    text = "abcdefghij" * 50  # 500 chars, no natural separators
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # The tail of chunk[i] (last `overlap` chars) should equal the head of chunk[i+1].
    for i in range(len(chunks) - 1):
        tail = chunks[i][-20:]
        head = chunks[i + 1][:20]
        assert tail == head


def test_hard_character_split_fallback_for_text_with_no_separators():
    # A single giant "word" with no spaces/newlines/periods at all.
    text = "x" * 350
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 4
    assert all(len(c) <= 100 for c in chunks)


def test_default_parameters_are_used_when_not_specified():
    text = "word " * 1000  # long enough to require the defaults to kick in
    chunks = recursive_character_split(text)
    assert len(chunks) > 1
    assert all(len(c) <= DEFAULT_CHUNK_SIZE for c in chunks)


# ---------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------


def test_chunk_document_returns_empty_list_for_blank_text():
    assert chunk_document(doc_id=1, category="sci.space", text="") == []
    assert chunk_document(doc_id=1, category="sci.space", text=None) == []


def test_chunk_document_assigns_stable_unique_chunk_ids():
    text = "Sentence. " * 200
    records = chunk_document(
        doc_id=42, category="rec.autos", text=text, chunk_size=100, chunk_overlap=10
    )
    assert len(records) > 1
    ids = [r["chunk_id"] for r in records]
    assert len(ids) == len(set(ids))  # all unique
    assert all(r["doc_id"] == 42 for r in records)
    assert all(r["category"] == "rec.autos" for r in records)
    assert ids == [f"42_{i}" for i in range(len(records))]


def test_chunk_document_records_have_expected_fields():
    records = chunk_document(
        doc_id="doc_a", category="talk.politics", text="Short text."
    )
    assert len(records) == 1
    record = records[0]
    for field in (
        "chunk_id",
        "doc_id",
        "category",
        "chunk_index",
        "chunk_text",
        "chunk_length",
        "num_chunks_in_doc",
    ):
        assert field in record
    assert record["chunk_length"] == len(record["chunk_text"])
    assert record["num_chunks_in_doc"] == 1


# ---------------------------------------------------------------------
# chunk_dataframe
# ---------------------------------------------------------------------


def test_chunk_dataframe_basic():
    df = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "category": ["sci.space", "sci.med", "rec.sport.hockey"],
            "clean_text": [
                "A reasonably short document about space exploration.",
                "",  # became empty after cleaning — should be skipped, not crash
                "Sentence one. " * 100,  # long enough to require multiple chunks
            ],
        }
    )
    result = chunk_dataframe(
        df, text_column="clean_text", chunk_size=200, chunk_overlap=20
    )

    assert isinstance(result, pd.DataFrame)
    # doc_id=2 had empty text -> contributes zero rows.
    assert 2 not in result["doc_id"].values
    # doc_id=1 is short -> exactly one chunk.
    assert (result["doc_id"] == 1).sum() == 1
    # doc_id=3 is long -> more than one chunk.
    assert (result["doc_id"] == 3).sum() > 1
    # chunk_id must be unique across the whole result.
    assert result["chunk_id"].is_unique


def test_chunk_dataframe_handles_all_empty_documents():
    df = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "category": ["misc.forsale", "misc.forsale"],
            "clean_text": ["", "   "],
        }
    )
    result = chunk_dataframe(df, text_column="clean_text")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_chunk_dataframe_preserves_category_metadata():
    df = pd.DataFrame(
        {
            "doc_id": [10],
            "category": ["comp.graphics"],
            "clean_text": ["A document about rendering pipelines and graphics cards."],
        }
    )
    result = chunk_dataframe(df, text_column="clean_text")
    assert result.iloc[0]["category"] == "comp.graphics"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
