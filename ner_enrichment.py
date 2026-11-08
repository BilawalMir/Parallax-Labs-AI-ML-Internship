"""
ner_enrichment.py
------------------
Stage 2 (Task 4): Named Entity Recognition over the chunked corpus,
writing entity metadata back onto each chunk in ChromaDB for filtered
retrieval (e.g. "only chunks mentioning Israel", "only chunks mentioning
NASA").

Model: spaCy en_core_web_sm. Chosen over a sentiment-analysis approach
because this corpus (20 Newsgroups) skews informational/discussion-heavy
in most categories (comp.*, sci.*, misc.forsale) rather than uniformly
opinionated, so entity mentions are a more broadly useful retrieval filter
across the whole corpus than a sentiment score would be. NER also isn't
covered by torch (spaCy's default small pipeline uses its own thinc
backend), which matters in constrained environments.

Entity types stored (out of spaCy's ~18 default labels): PERSON, ORG, GPE
(countries/cities/states), NORP (nationalities/religious/political
groups) -- chosen because this corpus is heavy on exactly this content
(FBI/NASA/NSA as ORG, Israel/Gaza as GPE, Israeli/Arab/Jewish as NORP,
named individuals as PERSON). Types like DATE, CARDINAL, ORDINAL, MONEY
are extracted by spaCy but not stored -- they're rarely what someone
filtering a Q&A/RAG retrieval system would search by.

Edge cases handled explicitly (per task requirements):
- Short chunks (< MIN_CHARS_FOR_NER characters): too little context for
  reliable entity recognition, skipped with ner_skipped_reason="too_short".
- Garbled / non-linguistic chunks (uuencoded binary attachments that leaked
  into the text corpus -- a real, confirmed issue in this dataset, see
  topic_modeling.py's topic 4/6 findings): detected via a cheap heuristic
  (ratio of alphabetic+whitespace characters to total length). Real English
  text -- even jargon-heavy technical text -- sits above ~0.9; the uuencoded
  chunks sit around ~0.5. Skipped with ner_skipped_reason="garbled_text"
  rather than letting spaCy emit meaningless entities on line noise.
- Jargon (technical terms, product/protocol names): not filtered out --
  spaCy still runs on these chunks. Accuracy impact is measured and
  documented in evaluate_ner_accuracy.py rather than silently patched;
  domain jargon is a known spaCy weak spot (e.g. mislabeling a product
  name as WORK_OF_ART) and belongs in the accuracy write-up, not a
  hard-coded fix.

Usage:
    python ner_enrichment.py
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

import chromadb

from topic_modeling import fetch_all_chunks, update_metadata

MIN_CHARS_FOR_NER = 30
ALPHA_SPACE_RATIO_THRESHOLD = 0.75  # below this, treat chunk as non-linguistic/garbled

SPACY_MODEL_NAME = "en_core_web_sm"

# spaCy label -> Chroma metadata field name. Only these four are stored;
# everything else spaCy detects (DATE, MONEY, CARDINAL, ...) is discarded.
ENTITY_TYPE_TO_FIELD = {
    "PERSON": "ner_persons",
    "ORG": "ner_orgs",
    "GPE": "ner_locations",
    "NORP": "ner_groups",
}

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "ner_enrichment_perf.csv")


# --------------------------------------------------------------------------
# Pure logic (no spaCy import) -- unit-testable without the model installed
# --------------------------------------------------------------------------

def alpha_space_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic or whitespace. Real
    English text (including jargon) sits around 0.9-1.0; uuencoded/binary
    junk that leaked into a few chunks sits around 0.5."""
    if not text:
        return 0.0
    alpha_space = sum(1 for c in text if c.isalpha() or c.isspace())
    return alpha_space / len(text)


def partition_for_ner(
    ids: list[str],
    texts: list[str],
    min_chars: int = MIN_CHARS_FOR_NER,
    min_alpha_ratio: float = ALPHA_SPACE_RATIO_THRESHOLD,
) -> tuple[dict, dict]:
    """
    Splits chunks into (usable, skipped). `skipped` entries carry a reason
    ("too_short" or "garbled_text") so it's traceable in the metadata why
    a chunk has no entity data, rather than it looking like NER just
    silently found nothing.
    """
    usable = {"ids": [], "texts": []}
    skipped = {"ids": [], "texts": [], "reasons": []}

    for cid, text in zip(ids, texts):
        if not isinstance(text, str) or len(text.strip()) < min_chars:
            skipped["ids"].append(cid)
            skipped["texts"].append(text)
            skipped["reasons"].append("too_short")
        elif alpha_space_ratio(text) < min_alpha_ratio:
            skipped["ids"].append(cid)
            skipped["texts"].append(text)
            skipped["reasons"].append("garbled_text")
        else:
            usable["ids"].append(cid)
            usable["texts"].append(text)

    return usable, skipped


def entities_from_spacy_doc(doc) -> dict[str, list[str]]:
    """
    Extracts the entity types we care about from a spaCy Doc, deduped
    (case-insensitive) while preserving first-seen order and original
    casing. Works against anything exposing `.ents` with `.label_`/`.text`
    (real spaCy Doc, or a test double), so this stays unit-testable
    without spaCy installed.
    """
    result: dict[str, list[str]] = {field: [] for field in ENTITY_TYPE_TO_FIELD.values()}
    seen: dict[str, set] = {field: set() for field in ENTITY_TYPE_TO_FIELD.values()}

    for ent in doc.ents:
        field = ENTITY_TYPE_TO_FIELD.get(ent.label_)
        if field is None:
            continue
        key = ent.text.strip().lower()
        if not key or key in seen[field]:
            continue
        seen[field].add(key)
        result[field].append(ent.text.strip())

    return result


def build_ner_metadata_updates(
    usable_ids: list[str],
    usable_entities: list[dict[str, list[str]]],
    skipped_ids: list[str],
    skipped_reasons: list[str],
) -> tuple[list[str], list[dict]]:
    """
    Builds the (ids, metadatas) pair for collection.update(). Chroma
    metadata values must be str/int/float/bool (no lists), so each entity
    list is joined into a single comma-separated string; empty -> "".
    """
    all_ids: list[str] = []
    all_metas: list[dict] = []

    for cid, entities in zip(usable_ids, usable_entities):
        meta = {field: ", ".join(values) for field, values in entities.items()}
        meta["ner_entity_count"] = sum(len(v) for v in entities.values())
        all_ids.append(cid)
        all_metas.append(meta)

    for cid, reason in zip(skipped_ids, skipped_reasons):
        meta = {field: "" for field in ENTITY_TYPE_TO_FIELD.values()}
        meta["ner_entity_count"] = 0
        meta["ner_skipped_reason"] = reason
        all_ids.append(cid)
        all_metas.append(meta)

    return all_ids, all_metas


def log_performance(stats: dict, note: str = ""):
    """Append a row of NER run stats to logs/ner_enrichment_perf.csv."""
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "num_chunks_total", "num_chunks_processed",
                "num_chunks_too_short", "num_chunks_garbled",
                "total_entities_found", "process_time_sec", "note",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            stats["num_chunks_total"],
            stats["num_chunks_processed"],
            stats["num_chunks_too_short"],
            stats["num_chunks_garbled"],
            stats["total_entities_found"],
            stats["process_time_sec"],
            note,
        ])
    print(f"Logged performance to {LOG_FILE}")


# --------------------------------------------------------------------------
# spaCy pipeline (lazy import -- same pattern as bertopic in topic_modeling.py)
# --------------------------------------------------------------------------

def run_ner_batch(
    texts: list[str],
    model_name: str = SPACY_MODEL_NAME,
    batch_size: int = 256,
    progress_every: int = 2000,
) -> list[dict]:
    """Runs spaCy's nlp.pipe() over all usable texts and extracts entities
    per chunk. nlp.pipe() (not calling nlp() per text in a loop) batches
    the tokenizer/tagger/parser/NER pipeline internally -- meaningfully
    faster than a plain Python loop over 35k+ chunks.

    Disables tagger/parser/attribute_ruler/lemmatizer: en_core_web_sm's ner
    component listens directly to tok2vec, not to the tagger or parser's
    output, so disabling them doesn't change entity results at all (verified
    side-by-side: identical entities with and without) -- it just skips
    dependency parsing, which is by far the slowest part of the pipeline and
    entirely unused here.

    Prints progress every `progress_every` chunks. On the full ~32k-chunk
    corpus this step legitimately takes several minutes even with the
    trimmed pipeline -- without any output in that window it looks
    identical to a hang, which is exactly what happened on the first run.
    Same reasoning as fetch_all_chunks's "fetched X/Y" prints.
    """
    import spacy

    nlp = spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    results = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size), start=1):
        results.append(entities_from_spacy_doc(doc))
        if i % progress_every == 0 or i == len(texts):
            print(f"  processed {i}/{len(texts)} chunks for NER...")
    return results


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_ner_enrichment(db_path: str = "./chroma_db", collection_name: str = "documents"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    ids, texts, _metas = fetch_all_chunks(collection)

    usable, skipped = partition_for_ner(ids, texts)
    num_too_short = skipped["reasons"].count("too_short")
    num_garbled = skipped["reasons"].count("garbled_text")
    print(f"{len(usable['ids'])} chunks usable for NER, "
          f"{num_too_short} too short, {num_garbled} garbled -- skipped.")

    start = time.perf_counter()
    usable_entities = run_ner_batch(usable["texts"]) if usable["ids"] else []
    process_time = time.perf_counter() - start

    total_entities = sum(sum(len(v) for v in e.values()) for e in usable_entities)
    print(f"Extracted {total_entities} entities across {len(usable['ids'])} chunks in {process_time:.1f}s.")

    all_ids, all_metas = build_ner_metadata_updates(
        usable["ids"], usable_entities, skipped["ids"], skipped["reasons"],
    )
    update_metadata(collection, all_ids, all_metas)

    log_performance({
        "num_chunks_total": len(ids),
        "num_chunks_processed": len(usable["ids"]),
        "num_chunks_too_short": num_too_short,
        "num_chunks_garbled": num_garbled,
        "total_entities_found": total_entities,
        "process_time_sec": round(process_time, 2),
    }, note="full corpus run")

    return all_ids, all_metas


if __name__ == "__main__":
    run_ner_enrichment()
