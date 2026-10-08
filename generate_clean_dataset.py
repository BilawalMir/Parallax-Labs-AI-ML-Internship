"""
generate_clean_dataset.py
--------------------------
Stage 4 (final step): Applies text_cleaning.py to the full raw dataset
(data_raw.csv from Stage 2), producing:

  - processed_dataset.csv   : cleaned dataset, ready for chunking/embedding
  - data_quality_report.md  : before/after quality report

Requires data_raw.csv to already exist (run load_and_validate_dataset.py first).

Run:
    python generate_clean_dataset.py
"""

import sys
import pandas as pd
from tqdm import tqdm

from text_cleaning import clean_text, detect_language, is_valid_text

RAW_INPUT_PATH = "data_raw.csv"
PROCESSED_OUTPUT_PATH = "processed_dataset.csv"
QUALITY_REPORT_PATH = "data_quality_report.md"


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the cleaning pipeline and language detection to every document.
    Uses tqdm to show progress since this runs over 18,000+ documents.
    """
    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df["text"].progress_apply(clean_text)

    tqdm.pandas(desc="Detecting language")
    df["language"] = df["clean_text"].progress_apply(detect_language)

    df["original_length"] = df["text"].apply(
        lambda t: len(t) if isinstance(t, str) else 0
    )
    df["clean_length"] = df["clean_text"].apply(len)

    return df


def build_report(df: pd.DataFrame) -> str:
    total_docs = len(df)

    # Docs that became empty after cleaning (all content was noise/headers)
    became_empty = int((df["clean_length"] == 0).sum())

    # Length reduction stats
    avg_original_len = round(df["original_length"].mean(), 1)
    avg_clean_len = round(df["clean_length"].mean(), 1)
    pct_reduction = (
        round(100 * (1 - avg_clean_len / avg_original_len), 1)
        if avg_original_len
        else 0
    )

    # Language distribution (top 10, since 20 Newsgroups is mostly English —
    # this surfaces any real mixed-language content or misdetections)
    lang_counts = df["language"].value_counts().head(10).to_dict()

    # Duplicate check post-cleaning (headers/signatures removed can sometimes
    # reveal near-duplicates that weren't exact matches in the raw text)
    post_clean_duplicates = int(df.duplicated(subset="clean_text").sum())

    # Very short cleaned docs (<20 chars, but not empty) — candidates for exclusion downstream
    short_mask = (df["clean_length"] > 0) & (df["clean_length"] < 20)
    very_short_after_cleaning = int(short_mask.sum())

    lines = [
        "# Data Quality Report — Post-Cleaning\n",
        f"**Total documents processed:** {total_docs}\n",
        "## Cleaning Impact\n",
        f"- Average original length: **{avg_original_len} characters**",
        f"- Average cleaned length: **{avg_clean_len} characters**",
        f"- Average size reduction: **{pct_reduction}%** (headers, quotes, signatures, HTML, URLs, emails, emoji removed)",
        f"- Documents that became empty after cleaning (pure noise/headers): **{became_empty}**",
        f"- Very short cleaned documents (<20 chars): **{very_short_after_cleaning}**",
        f"- Duplicate documents after cleaning (not caught by raw-text duplicate check): **{post_clean_duplicates}**",
        "",
        "## Language Distribution (top 10, post-cleaning)\n",
    ]
    for lang, count in lang_counts.items():
        pct = round(100 * count / total_docs, 2)
        lines.append(f"- {lang}: {count} ({pct}%)")

    lines += [
        "",
        "## Notes\n",
        "- `language == 'unknown'` means the cleaned text was empty or too short/ambiguous "
        "for reliable detection — expected for near-empty posts after header/quote removal.",
        "- Documents that became empty after cleaning are still present in "
        "`processed_dataset.csv` (not silently dropped) so downstream steps can decide "
        "how to handle them — flagging this as a decision point rather than assuming "
        "they should be deleted.",
    ]

    return "\n".join(lines)


def main() -> int:
    try:
        df = pd.read_csv(RAW_INPUT_PATH)
    except FileNotFoundError:
        print(
            f"ERROR: {RAW_INPUT_PATH} not found. Run load_and_validate_dataset.py first."
        )
        return 1

    print(f"Loaded {len(df)} raw documents from {RAW_INPUT_PATH}")

    df = process_dataset(df)

    df.to_csv(PROCESSED_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nProcessed dataset saved to {PROCESSED_OUTPUT_PATH}")

    report = build_report(df)
    with open(QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Data quality report saved to {QUALITY_REPORT_PATH}")

    print("\n" + "=" * 60)
    print("DONE — Stage 4 complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
