"""
chunking.py
-----------
Stage 1 (Task 2): Recursive text chunking strategy for the cleaned
20 Newsgroups dataset (processed_dataset.csv from Task 1).

Chunking strategy
------------------
Uses **recursive character splitting**: text is split by trying a list of
separators in order of "semantic size" (biggest/most natural break first),
falling back to smaller separators only where needed:

    1. "\n\n"  -> paragraph breaks
    2. "\n"    -> line breaks
    3. ". "    -> sentence breaks
    4. " "     -> word breaks
    5. ""      -> hard character split (last resort, guarantees termination)

For each candidate separator, the text is split into pieces, and pieces are
greedily packed together until adding the next piece would exceed
CHUNK_SIZE. If a single piece is still bigger than CHUNK_SIZE (e.g. one
giant paragraph with no periods), the next separator down the list is tried
recursively on that piece. This is the same core idea as LangChain's
RecursiveCharacterTextSplitter, reimplemented here directly so there's no
extra dependency.

Overlap: consecutive chunks share CHUNK_OVERLAP characters of context
(taken from the end of the previous chunk), which helps semantic search
retrieve chunks even when the relevant sentence sits right at a chunk
boundary.

Defaults chosen for this task: CHUNK_SIZE=1000, CHUNK_OVERLAP=100
(~10% overlap), which is a reasonable balance for 20 Newsgroups-style
posts (short-to-medium length, mostly plain text).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Splits text on a separator, keeping the separator attached to each
    piece (except possibly the last) so re-joining doesn't lose meaning."""
    if separator == "":
        # Last resort: split into individual characters.
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to all pieces except the final one.
    rejoined = [p + separator for p in parts[:-1]] + [parts[-1]]
    # Drop any empty strings produced by the split (e.g. leading/trailing separator).
    return [p for p in rejoined if p != ""]


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merges small (atomic, already <= chunk_size) pieces into
    chunks up to chunk_size, carrying `chunk_overlap` characters of trailing
    context into the next chunk.

    Invariant maintained: `current` is never allowed to exceed chunk_size.
    The overlap tail from the previous chunk is only prepended to the next
    chunk if it still fits alongside the next piece — otherwise the next
    chunk simply starts fresh with that piece (this only happens for very
    large individual pieces sitting right at the chunk_size limit).
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            if len(tail) + len(piece) <= chunk_size:
                current = tail + piece
            else:
                current = piece

    if current:
        chunks.append(current)

    return chunks


def recursive_character_split(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively splits `text` into chunks of at most ~chunk_size characters,
    with chunk_overlap characters shared between consecutive chunks.

    Implementation is two separate passes:
    1. `_split_into_pieces` recursively breaks the text down into small,
       "atomic" pieces (each already <= chunk_size wherever possible) —
       this pass never merges anything, so nothing can compound over
       chunk_size across recursion levels.
    2. `_merge_pieces` does a single greedy merge pass over those atomic
       pieces to pack them into chunk_size-sized chunks with overlap.

    Edge cases handled:
    - None / non-string input -> returns []
    - empty or whitespace-only string -> returns []
    - text shorter than chunk_size -> returns a single chunk (the whole text)
    - chunk_overlap >= chunk_size -> treated as chunk_overlap = 0 (avoids
      infinite loops / non-progress)
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = 0

    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = DEFAULT_SEPARATORS

    pieces = _split_into_pieces(text, chunk_size, separators)
    return _merge_pieces(pieces, chunk_size, chunk_overlap)


def _split_into_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Pure recursive split: breaks text into pieces each <= chunk_size
    (where at all possible), WITHOUT merging or applying overlap. The final
    separator in the list is "" (character split), which always guarantees
    every piece ends up <= chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Should not normally be reached (separators ends in ""), but guard
        # with a hard character split just in case.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    remaining_separators = separators[1:]
    raw_pieces = _split_on_separator(text, separator)

    result: list[str] = []
    for piece in raw_pieces:
        if len(piece) > chunk_size:
            result.extend(_split_into_pieces(piece, chunk_size, remaining_separators))
        else:
            result.append(piece)
    return result


def chunk_document(
    doc_id, category: str, text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunks a single document and returns a list of chunk records (dicts),
    each with a stable, unique chunk_id derived from doc_id + chunk_index.
    Returns [] if the document has no usable text (handles the same edge
    cases as recursive_character_split).
    """
    pieces = recursive_character_split(text, chunk_size, chunk_overlap)
    records = []
    for i, piece in enumerate(pieces):
        records.append({
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "category": category,
            "chunk_index": i,
            "chunk_text": piece,
            "chunk_length": len(piece),
            "num_chunks_in_doc": len(pieces),
        })
    return records


def chunk_dataframe(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    doc_id_column: str = "doc_id",
    category_column: str = "category",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> pd.DataFrame:
    """
    Applies chunk_document() across an entire DataFrame (e.g.
    processed_dataset.csv) and returns a flat DataFrame of chunks, one row
    per chunk, ready for embedding.

    Documents that produce zero chunks (empty/whitespace-only clean_text —
    see data_quality_report.md from Task 1) are simply skipped; nothing
    crashes on them.
    """
    all_records: list[dict] = []
    for row in df.itertuples(index=False):
        doc_id = getattr(row, doc_id_column)
        category = getattr(row, category_column) if hasattr(row, category_column) else None
        text = getattr(row, text_column)
        all_records.extend(
            chunk_document(doc_id, category, text, chunk_size, chunk_overlap)
        )

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    # Small smoke test / demo when run directly.
    sample = (
        "Newsgroups are one of the oldest forms of online discussion. "
        "They predate the modern web by over a decade.\n\n"
        "Each newsgroup is dedicated to a specific topic, from computer "
        "hardware to politics to recreational hobbies. Users post messages "
        "and others reply, forming threaded discussions.\n\n"
    ) * 5
    chunks = recursive_character_split(sample)
    print(f"Sample text length: {len(sample)} chars -> {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c)} chars")