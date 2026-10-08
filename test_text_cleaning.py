"""
test_text_cleaning.py
----------------------
Stage 4: Unit tests for text_cleaning.py

Run:
    pytest test_text_cleaning.py -v
"""

import pytest
from text_cleaning import (
    is_valid_text,
    fix_encoding,
    remove_newsgroup_headers,
    remove_quoted_lines,
    remove_signature,
    remove_html_tags,
    remove_urls,
    remove_email_addresses,
    remove_emoji,
    normalize_whitespace,
    detect_language,
    clean_text,
)

# ---------------------------------------------------------------------------
# is_valid_text
# ---------------------------------------------------------------------------


def test_is_valid_text_none():
    assert is_valid_text(None) is False


def test_is_valid_text_empty_string():
    assert is_valid_text("") is False


def test_is_valid_text_whitespace_only():
    assert is_valid_text("   \n\t  ") is False


def test_is_valid_text_non_string_input():
    assert is_valid_text(12345) is False
    assert is_valid_text(["not", "a", "string"]) is False


def test_is_valid_text_normal_string():
    assert is_valid_text("hello world") is True


# ---------------------------------------------------------------------------
# fix_encoding
# ---------------------------------------------------------------------------


def test_fix_encoding_none_returns_empty():
    assert fix_encoding(None) == ""


def test_fix_encoding_empty_returns_empty():
    assert fix_encoding("") == ""


def test_fix_encoding_fixes_mojibake():
    broken = "This textâ€™s got mojibake"
    fixed = fix_encoding(broken)
    assert "â€™" not in fixed


def test_fix_encoding_leaves_clean_text_unchanged():
    clean = "This is already clean text."
    assert fix_encoding(clean) == clean


# ---------------------------------------------------------------------------
# remove_newsgroup_headers
# ---------------------------------------------------------------------------


def test_remove_newsgroup_headers_strips_header_block():
    text = "From: a@b.com\nSubject: Test\nLines: 3\n\nActual message body here."
    result = remove_newsgroup_headers(text)
    assert "From:" not in result
    assert "Subject:" not in result
    assert "Actual message body here." in result


def test_remove_newsgroup_headers_none_returns_empty():
    assert remove_newsgroup_headers(None) == ""


def test_remove_newsgroup_headers_no_headers_present():
    text = "Just a plain message with no headers at all."
    assert remove_newsgroup_headers(text) == text


# ---------------------------------------------------------------------------
# remove_quoted_lines
# ---------------------------------------------------------------------------


def test_remove_quoted_lines_removes_quotes():
    text = "Normal line.\n> This is quoted.\nAnother normal line."
    result = remove_quoted_lines(text)
    assert "> This is quoted." not in result
    assert "Normal line." in result
    assert "Another normal line." in result


def test_remove_quoted_lines_none_returns_empty():
    assert remove_quoted_lines(None) == ""


def test_remove_quoted_lines_no_quotes_present():
    text = "No quoted lines here."
    assert remove_quoted_lines(text) == text


# ---------------------------------------------------------------------------
# remove_signature
# ---------------------------------------------------------------------------


def test_remove_signature_strips_after_dash_dash():
    text = "Main message.\n--\nJohn Doe\njohn@example.com"
    result = remove_signature(text)
    assert "Main message." in result
    assert "John Doe" not in result


def test_remove_signature_none_returns_empty():
    assert remove_signature(None) == ""


def test_remove_signature_no_signature_present():
    text = "Just a message, no signature marker."
    assert remove_signature(text) == text


# ---------------------------------------------------------------------------
# remove_html_tags
# ---------------------------------------------------------------------------


def test_remove_html_tags_strips_tags():
    text = "This is <b>bold</b> and <i>italic</i>."
    result = remove_html_tags(text)
    assert "<b>" not in result
    assert "<i>" not in result
    assert "bold" in result
    assert "italic" in result


def test_remove_html_tags_none_returns_empty():
    assert remove_html_tags(None) == ""


# ---------------------------------------------------------------------------
# remove_urls
# ---------------------------------------------------------------------------


def test_remove_urls_removes_http_links():
    text = "Check this out: https://example.com/page and http://another.com"
    result = remove_urls(text)
    assert "https://" not in result
    assert "http://" not in result
    assert "Check this out:" in result


def test_remove_urls_none_returns_empty():
    assert remove_urls(None) == ""


# ---------------------------------------------------------------------------
# remove_email_addresses
# ---------------------------------------------------------------------------


def test_remove_email_addresses_removes_emails():
    text = "Contact me at someone@example.com for details."
    result = remove_email_addresses(text)
    assert "someone@example.com" not in result
    assert "Contact me at" in result


def test_remove_email_addresses_none_returns_empty():
    assert remove_email_addresses(None) == ""


# ---------------------------------------------------------------------------
# remove_emoji
# ---------------------------------------------------------------------------


def test_remove_emoji_removes_emoji_keeps_text():
    text = "Great card 😀 highly recommend! 👍"
    result = remove_emoji(text)
    assert "😀" not in result
    assert "👍" not in result
    assert "Great card" in result
    assert "highly recommend!" in result


def test_remove_emoji_none_returns_empty():
    assert remove_emoji(None) == ""


def test_remove_emoji_no_emoji_present():
    text = "No emoji here at all."
    assert remove_emoji(text) == text


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


def test_normalize_whitespace_collapses_spaces():
    text = "Too    many     spaces"
    assert normalize_whitespace(text) == "Too many spaces"


def test_normalize_whitespace_collapses_newlines():
    text = "Line one\n\n\n\nLine two"
    assert normalize_whitespace(text) == "Line one Line two"


def test_normalize_whitespace_trims_ends():
    text = "   leading and trailing spaces   "
    assert normalize_whitespace(text) == "leading and trailing spaces"


def test_normalize_whitespace_none_returns_empty():
    assert normalize_whitespace(None) == ""


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


def test_detect_language_english():
    text = "This is a normal English sentence about computer hardware and software."
    assert detect_language(text) == "en"


def test_detect_language_empty_returns_unknown():
    assert detect_language("") == "unknown"


def test_detect_language_none_returns_unknown():
    assert detect_language(None) == "unknown"


def test_detect_language_never_raises_on_short_ambiguous_text():
    # langdetect can raise LangDetectException internally on very short/
    # ambiguous text (e.g. a single punctuation character) — this must be
    # caught and turned into "unknown", never propagate as a crash.
    result = detect_language("...")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# clean_text (full pipeline)
# ---------------------------------------------------------------------------


def test_clean_text_none_returns_empty():
    assert clean_text(None) == ""


def test_clean_text_empty_returns_empty():
    assert clean_text("") == ""


def test_clean_text_whitespace_only_returns_empty():
    assert clean_text("   \n\t  ") == ""


def test_clean_text_non_string_returns_empty():
    assert clean_text(12345) == ""
    assert clean_text(["list", "not", "text"]) == ""


def test_clean_text_full_pipeline_on_realistic_messy_post():
    sample = (
        "From: someone@example.com\n"
        "Subject: Re: Best graphics card?\n"
        "Organization: Some University\n"
        "Lines: 15\n\n"
        "In article <12345@example.com>, another@example.com wrote:\n"
        "> I think the XYZ card is the best for gaming.\n"
        "> Check this out: http://example.com/review\n\n"
        "That's mostly true, but see <b>this link</b> too: https://another.com/page\n"
        "This card is great for the price 😀 <br> highly recommend!\n\n"
        "--\n"
        "John Doe\n"
        "someone@example.com\n"
    )
    result = clean_text(sample)

    # Headers, quotes, signature, HTML, URLs, emails, emoji should all be gone
    assert "From:" not in result
    assert "Subject:" not in result
    assert "> I think" not in result
    assert "John Doe" not in result
    assert "<b>" not in result
    assert "http://" not in result
    assert "https://" not in result
    assert "someone@example.com" not in result
    assert "😀" not in result

    # The actual message content should survive
    assert "mostly true" in result
    assert "highly recommend" in result


def test_clean_text_mixed_language_content_is_preserved():
    # Cleaning should not strip non-English text — only noise.
    text = "Ye card sach mein bohot acha hai for the price."
    result = clean_text(text)
    assert "sach mein" in result
    assert "acha hai" in result


def test_clean_text_is_idempotent_on_already_clean_text():
    # Running clean_text on already-clean text should not corrupt it further
    # (aside from whitespace normalization, which is safe to reapply).
    text = "This is already a clean, simple sentence."
    once = clean_text(text)
    twice = clean_text(once)
    assert once == twice
