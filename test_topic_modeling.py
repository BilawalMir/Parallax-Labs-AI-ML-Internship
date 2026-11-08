"""
test_topic_modeling.py
-----------------------
Unit tests for topic_modeling.py's pure-logic functions: triage (via
nlp_utils.is_usable_for_nlp), LDA fitting on tiny in-memory corpora, and
building/writing the Chroma metadata-update payload. These deliberately
avoid requiring bertopic/torch to be installed (that heavy import is lazy,
isolated inside run_bertopic()) so these tests can run in any environment,
same spirit as test_generate_answer.py's mocked tests needing no API key.

Run: python -m pytest test_topic_modeling.py -v
"""

import os
import tempfile

import pytest
from unittest.mock import MagicMock

from topic_modeling import (
    fetch_all_chunks,
    log_performance,
    run_lda,
    triage,
    update_metadata,
    write_topic_summary,
)

try:
    import bertopic  # noqa: F401
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False


# --------------------------------------------------------------------------
# fetch_all_chunks
# --------------------------------------------------------------------------

def test_fetch_all_chunks_paginates_until_exhausted():
    fake_collection = MagicMock()
    fake_collection.name = "documents"
    fake_collection.count.return_value = 3
    fake_collection.get.side_effect = [
        {"ids": ["c1", "c2"], "documents": ["doc1", "doc2"], "metadatas": [{}, {}]},
        {"ids": ["c3"], "documents": ["doc3"], "metadatas": [{}]},
    ]

    ids, docs, metas = fetch_all_chunks(fake_collection)

    assert ids == ["c1", "c2", "c3"]
    assert docs == ["doc1", "doc2", "doc3"]
    assert len(metas) == 3


def test_fetch_all_chunks_empty_collection():
    fake_collection = MagicMock()
    fake_collection.name = "documents"
    fake_collection.count.return_value = 0

    ids, docs, metas = fetch_all_chunks(fake_collection)

    assert ids == []
    assert docs == []
    assert metas == []


# --------------------------------------------------------------------------
# triage
# --------------------------------------------------------------------------

def test_triage_splits_usable_too_short_garbled():
    ids = ["c1", "c2", "c3"]
    docs = [
        "hi",  # too short
        "This is a perfectly normal, long enough sentence about graphics cards.",  # usable
        "3n9!@#812&*(zzz882--93kk",  # garbled (low vowel ratio / high digit ratio)
    ]

    usable_ids, usable_docs, too_short_ids, garbled_ids = triage(ids, docs)

    assert usable_ids == ["c2"]
    assert usable_docs == [docs[1]]
    assert too_short_ids == ["c1"]
    assert garbled_ids == ["c3"]


def test_triage_empty_input():
    usable_ids, usable_docs, too_short_ids, garbled_ids = triage([], [])
    assert usable_ids == []
    assert usable_docs == []
    assert too_short_ids == []
    assert garbled_ids == []


def test_triage_all_usable():
    ids = ["c1", "c2"]
    docs = [
        "This is a normal, long enough sentence about space and NASA missions.",
        "Another perfectly readable chunk of English text about encryption keys.",
    ]
    usable_ids, usable_docs, too_short_ids, garbled_ids = triage(ids, docs)
    assert usable_ids == ["c1", "c2"]
    assert too_short_ids == []
    assert garbled_ids == []


# --------------------------------------------------------------------------
# run_lda
# --------------------------------------------------------------------------

def test_run_lda_returns_assignment_per_doc():
    docs = [
        "graphics card driver video display monitor",
        "graphics card driver video display screen",
        "encryption key cipher security algorithm",
        "encryption key cipher security protocol",
    ] * 3  # repeat so CountVectorizer's min_df=2 has enough support

    assignments, topic_info = run_lda(docs, n_topics=2)

    assert len(assignments) == len(docs)
    assert set(assignments).issubset(set(topic_info.keys()))


def test_run_lda_topic_info_has_keywords_and_label():
    docs = [
        "graphics card driver video display monitor",
        "graphics card driver video display screen",
        "encryption key cipher security algorithm",
        "encryption key cipher security protocol",
    ] * 3

    _assignments, topic_info = run_lda(docs, n_topics=2)

    for info in topic_info.values():
        assert "keywords" in info
        assert "label" in info
        assert isinstance(info["keywords"], str)
        assert isinstance(info["label"], str)


def test_run_lda_defaults_to_15_topics_when_none_given():
    docs = [
        "graphics card driver video display monitor",
        "graphics card driver video display screen",
    ] * 5

    _assignments, topic_info = run_lda(docs, n_topics=None)
    # default n_topics=15, but LDA still only reports as many components as requested
    assert len(topic_info) == 15


# --------------------------------------------------------------------------
# run_bertopic (only when the optional heavy dependency is installed)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="bertopic not installed")
def test_run_bertopic_returns_assignment_per_doc():
    from topic_modeling import run_bertopic

    docs = [
        "graphics card driver video display monitor",
        "graphics card driver video display screen",
        "encryption key cipher security algorithm",
        "encryption key cipher security protocol",
    ] * 5

    assignments, topic_info = run_bertopic(docs, n_topics=None)
    assert len(assignments) == len(docs)
    assert isinstance(topic_info, dict)


# --------------------------------------------------------------------------
# update_metadata
# --------------------------------------------------------------------------

def test_update_metadata_batches_calls():
    fake_collection = MagicMock()
    ids = [f"c{i}" for i in range(5)]
    metas = [{"topic_id": i} for i in range(5)]

    update_metadata(fake_collection, ids, metas, update_batch=2)

    assert fake_collection.update.call_count == 3
    first_call_kwargs = fake_collection.update.call_args_list[0].kwargs
    assert first_call_kwargs["ids"] == ["c0", "c1"]


def test_update_metadata_empty_input_makes_no_calls():
    fake_collection = MagicMock()
    update_metadata(fake_collection, [], [])
    fake_collection.update.assert_not_called()


# --------------------------------------------------------------------------
# write_topic_summary
# --------------------------------------------------------------------------

def test_write_topic_summary_creates_file_with_topics():
    topic_info = {0: {"keywords": "space, nasa, orbit", "label": "space_nasa_orbit"}}
    assignments = [0, 0, 0]
    ids = ["c1", "c2", "c3"]
    docs = ["about space missions", "nasa launch update", "orbit mechanics discussion"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "topic_summary.md")
        write_topic_summary(topic_info, assignments, ids, docs, path=path)

        assert os.path.isfile(path)
        content = open(path, encoding="utf-8").read()
        assert "space_nasa_orbit" in content
        assert "space, nasa, orbit" in content


# --------------------------------------------------------------------------
# log_performance
# --------------------------------------------------------------------------

def test_log_performance_writes_header_once():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "logs", "topic_modeling_perf.csv")
        header = ["timestamp", "method", "total_chunks", "usable_chunks", "n_topics", "elapsed_seconds"]

        log_performance(path, ["t1", "lda", 10, 8, 3, 1.2], header)
        log_performance(path, ["t2", "lda", 10, 8, 3, 1.1], header)

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0].strip() == ",".join(header)
        assert len(lines) == 3  # header + 2 rows


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
