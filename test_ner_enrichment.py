"""
test_ner_enrichment.py
------------------------
Unit tests for ner_enrichment.py's pure-logic functions: the garbled-text
heuristic, length/garbled partitioning, entity extraction from a spaCy-Doc
-shaped object, and the Chroma metadata-update payload builder. Uses a
lightweight fake Doc/Span (not a real spaCy model) so these run without
downloading en_core_web_sm, same spirit as test_topic_modeling.py's fake
BERTopic model.

Run: python -m pytest test_ner_enrichment.py -v
"""

import pytest

from ner_enrichment import (
    alpha_space_ratio,
    build_ner_metadata_updates,
    entities_from_spacy_doc,
    partition_for_ner,
)


# --------------------------------------------------------------------------
# alpha_space_ratio
# --------------------------------------------------------------------------

def test_alpha_space_ratio_normal_text_is_high():
    text = "The Clipper Chip contains a classified encryption algorithm."
    assert alpha_space_ratio(text) > 0.9


def test_alpha_space_ratio_garbled_uuencoded_text_is_low():
    # a real excerpt style from the corpus's uuencoded-binary chunks
    garbled = "'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>' MAX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>"
    assert alpha_space_ratio(garbled) < 0.75


def test_alpha_space_ratio_empty_string():
    assert alpha_space_ratio("") == 0.0


# --------------------------------------------------------------------------
# partition_for_ner
# --------------------------------------------------------------------------

def test_partition_for_ner_splits_short_garbled_and_usable():
    ids = ["c1", "c2", "c3"]
    texts = [
        "short",  # too short
        "'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>'AX>' MAX>'AX>'AX>'AX>'AX>'AX>'AX>",  # garbled, long enough
        "The Clipper Chip contains a classified encryption algorithm called Skipjack.",  # usable
    ]

    usable, skipped = partition_for_ner(ids, texts, min_chars=30)

    assert usable["ids"] == ["c3"]
    assert skipped["ids"] == ["c1", "c2"]
    assert skipped["reasons"] == ["too_short", "garbled_text"]


def test_partition_for_ner_empty_input():
    usable, skipped = partition_for_ner([], [])
    assert usable["ids"] == []
    assert skipped["ids"] == []


def test_partition_for_ner_non_string_text_is_too_short():
    usable, skipped = partition_for_ner(["c1"], [None], min_chars=30)
    assert usable["ids"] == []
    assert skipped["reasons"] == ["too_short"]


# --------------------------------------------------------------------------
# entities_from_spacy_doc
# --------------------------------------------------------------------------

class _FakeSpan:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


def test_entities_from_spacy_doc_groups_by_type():
    doc = _FakeDoc([
        _FakeSpan("NASA", "ORG"),
        _FakeSpan("Neil Armstrong", "PERSON"),
        _FakeSpan("Israel", "GPE"),
        _FakeSpan("Israeli", "NORP"),
    ])
    result = entities_from_spacy_doc(doc)
    assert result["ner_orgs"] == ["NASA"]
    assert result["ner_persons"] == ["Neil Armstrong"]
    assert result["ner_locations"] == ["Israel"]
    assert result["ner_groups"] == ["Israeli"]


def test_entities_from_spacy_doc_ignores_unstored_types():
    # DATE/CARDINAL/etc are not in ENTITY_TYPE_TO_FIELD -- should be dropped silently
    doc = _FakeDoc([_FakeSpan("July 20, 1969", "DATE"), _FakeSpan("64", "CARDINAL")])
    result = entities_from_spacy_doc(doc)
    assert all(v == [] for v in result.values())


def test_entities_from_spacy_doc_dedupes_case_insensitively_preserving_first_casing():
    doc = _FakeDoc([_FakeSpan("Israel", "GPE"), _FakeSpan("israel", "GPE"), _FakeSpan("ISRAEL", "GPE")])
    result = entities_from_spacy_doc(doc)
    assert result["ner_locations"] == ["Israel"]  # only first occurrence kept


def test_entities_from_spacy_doc_no_entities():
    result = entities_from_spacy_doc(_FakeDoc([]))
    assert all(v == [] for v in result.values())


# --------------------------------------------------------------------------
# build_ner_metadata_updates
# --------------------------------------------------------------------------

def test_build_ner_metadata_updates_usable_chunk():
    entities = [{"ner_persons": ["Neil Armstrong"], "ner_orgs": ["NASA"], "ner_locations": [], "ner_groups": []}]
    all_ids, all_metas = build_ner_metadata_updates(["c1"], entities, [], [])

    assert all_ids == ["c1"]
    assert all_metas[0]["ner_persons"] == "Neil Armstrong"
    assert all_metas[0]["ner_orgs"] == "NASA"
    assert all_metas[0]["ner_locations"] == ""
    assert all_metas[0]["ner_entity_count"] == 2


def test_build_ner_metadata_updates_skipped_chunk():
    all_ids, all_metas = build_ner_metadata_updates([], [], ["c2"], ["garbled_text"])

    assert all_ids == ["c2"]
    assert all_metas[0]["ner_entity_count"] == 0
    assert all_metas[0]["ner_skipped_reason"] == "garbled_text"
    assert all_metas[0]["ner_persons"] == ""


def test_build_ner_metadata_updates_combines_both_partitions():
    entities = [{"ner_persons": ["Guido van Rossum"], "ner_orgs": [], "ner_locations": [], "ner_groups": []}]
    all_ids, all_metas = build_ner_metadata_updates(["c1"], entities, ["c2"], ["too_short"])

    assert all_ids == ["c1", "c2"]
    assert len(all_metas) == 2
    assert "ner_skipped_reason" not in all_metas[0]
    assert all_metas[1]["ner_skipped_reason"] == "too_short"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
