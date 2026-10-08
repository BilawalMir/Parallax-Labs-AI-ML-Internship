"""
text_cleaning.py
-----------------
Stage 3: Robust text cleaning functions for the 20 Newsgroups dataset.

Each function does ONE job and is independently testable (see
test_text_cleaning.py). clean_text() composes them into a single pipeline.

Design decisions (flagged explicitly, not assumed silently):
  - Functions never raise on None / empty / non-string input — they return
    a safe empty string instead, since a crash on one bad row would kill
    a run over 18,000+ documents.
  - Case and punctuation are NOT stripped/lowercased by default, because
    this dataset is being prepared for embeddings later, and embedding
    models generally perform better on natural casing/punctuation than on
    aggressively normalized text. If a downstream task needs lowercasing,
    it can be added as an explicit extra step — flagging this assumption
    rather than silently lowercasing everything.
"""

import re
import ftfy
import emoji
from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------


def is_valid_text(text) -> bool:
    """Returns False for None, non-strings, or empty/whitespace-only text."""
    if text is None:
        return False
    if not isinstance(text, str):
        return False
    if text.strip() == "":
        return False
    return True


def fix_encoding(text: str) -> str:
    """Fixes mojibake / broken encoding artifacts (e.g. â€™ -> ’) using ftfy."""
    if not is_valid_text(text):
        return ""
    return ftfy.fix_text(text)


def remove_newsgroup_headers(text: str) -> str:
    """
    Strips the email-style header block that 20 Newsgroups posts start with
    (From:, Subject:, Organization:, Lines:, NNTP-Posting-Host:, etc.).
    Headers are a contiguous block of "Key: value" lines at the top of the
    post, ending at the first blank line or first non-header line.
    """
    if not is_valid_text(text):
        return ""

    lines = text.split("\n")
    header_pattern = re.compile(r"^[A-Za-z\-]+:\s?.*$")
    body_start = 0

    for i, line in enumerate(lines):
        if header_pattern.match(line.strip()) or line.strip() == "":
            body_start = i + 1
        else:
            break

    return "\n".join(lines[body_start:])


def remove_quoted_lines(text: str) -> str:
    """Removes quoted-reply lines (lines starting with '>', common in Usenet posts)."""
    if not is_valid_text(text):
        return ""

    lines = text.split("\n")
    kept = [line for line in lines if not line.strip().startswith(">")]
    return "\n".join(kept)


def remove_signature(text: str) -> str:
    """
    Removes trailing signature blocks. Usenet convention: a line containing
    exactly '--' (optionally with trailing space) marks the start of a
    signature, and everything after it is dropped.
    """
    if not is_valid_text(text):
        return ""

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "--":
            return "\n".join(lines[:i])
    return text


def remove_html_tags(text: str) -> str:
    """Strips HTML/XML-style tags, e.g. <b>bold</b> -> bold."""
    if not is_valid_text(text):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Removes http(s) URLs and bare www.* links."""
    if not is_valid_text(text):
        return ""
    url_pattern = re.compile(r"(https?://\S+|www\.\S+)")
    return url_pattern.sub(" ", text)


def remove_email_addresses(text: str) -> str:
    """Removes email addresses, e.g. someone@example.com."""
    if not is_valid_text(text):
        return ""
    email_pattern = re.compile(r"\S+@\S+\.\S+")
    return email_pattern.sub(" ", text)


def remove_emoji(text: str) -> str:
    """Removes emoji characters, keeping surrounding text intact."""
    if not is_valid_text(text):
        return ""
    return emoji.replace_emoji(text, replace=" ")


def normalize_whitespace(text: str) -> str:
    """Collapses repeated whitespace/newlines into single spaces and trims ends."""
    if not is_valid_text(text):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """
    Detects the dominant language of a text. Returns 'unknown' for empty
    text or text too short/ambiguous for langdetect to classify reliably
    (it raises LangDetectException in those cases — caught here so a single
    bad row never crashes a full-dataset run).
    """
    if not is_valid_text(text):
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Runs the full cleaning pipeline in order:
      1. Fix encoding artifacts
      2. Remove newsgroup headers
      3. Remove quoted-reply lines
      4. Remove signature block
      5. Remove HTML tags
      6. Remove URLs
      7. Remove email addresses
      8. Remove emoji
      9. Normalize whitespace

    Returns "" for any input that is None, non-string, or empty/whitespace-only.
    """
    if not is_valid_text(text):
        return ""

    cleaned = fix_encoding(text)
    cleaned = remove_newsgroup_headers(cleaned)
    cleaned = remove_quoted_lines(cleaned)
    cleaned = remove_signature(cleaned)
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_email_addresses(cleaned)
    cleaned = remove_emoji(cleaned)
    cleaned = normalize_whitespace(cleaned)

    return cleaned
