"""
load_and_validate_dataset.py
-----------------------------
Stage 2: Acquire a large real-world text dataset (20 Newsgroups, ~18,800 docs)
and validate it before any cleaning happens.

Validation performed:
  1. Null / missing text checks
  2. Empty-string (whitespace-only) checks
  3. Duplicate document checks
  4. Encoding issue checks (can every doc be encoded/decoded as UTF-8 safely?)
  5. Document length distribution (catches suspiciously short/long docs)

Outputs:
  - data_raw.csv                 : the raw dataset as a DataFrame (doc_id, text, category)
  - data_validation_summary.md   : a human-readable validation report

Run:
    python load_and_validate_dataset.py
"""

import sys
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

RAW_OUTPUT_PATH = "data_raw.csv"
VALIDATION_REPORT_PATH = "data_validation_summary.md"


def acquire_dataset() -> pd.DataFrame:
    """
    Downloads the 20 Newsgroups dataset (train + test combined) and returns
    it as a DataFrame with columns: doc_id, text, category.

    NOTE: headers/footers/quotes are intentionally KEPT (not stripped) here,
    because Stage 3 (text_cleaning.py) needs real-world noise to clean —
    stripping it now would make the cleaning-function edge cases artificial.
    """
    print("Downloading 20 Newsgroups dataset (train + test)...")
    bunch = fetch_20newsgroups(
        subset="all",  # combines train + test => ~18,846 documents
        remove=(),  # keep headers/footers/quotes as real-world noise
        shuffle=True,
        random_state=42,
    )

    df = pd.DataFrame(
        {
            "doc_id": range(len(bunch.data)),
            "text": bunch.data,
            "category": [bunch.target_names[t] for t in bunch.target],
        }
    )

    print(f"Loaded {len(df)} documents across {df['category'].nunique()} categories.")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Runs all validation checks and returns a dict of results."""
    results = {}

    # 1. Null checks
    null_count = df["text"].isna().sum()
    results["null_count"] = int(null_count)

    # 2. Empty / whitespace-only checks (only meaningful on non-null rows)
    non_null_text = df["text"].dropna()
    empty_mask = non_null_text.str.strip().str.len() == 0
    results["empty_or_whitespace_count"] = int(empty_mask.sum())

    # 3. Duplicate checks (exact duplicate document text)
    duplicate_count = int(df.duplicated(subset="text").sum())
    results["duplicate_count"] = duplicate_count

    # 4. Encoding checks — try encode/decode as UTF-8, flag anything that fails
    def has_encoding_issue(text):
        if not isinstance(text, str):
            return False
        try:
            text.encode("utf-8").decode("utf-8")
            return False
        except (UnicodeEncodeError, UnicodeDecodeError):
            return True

    encoding_issues = non_null_text.apply(has_encoding_issue)
    results["encoding_issue_count"] = int(encoding_issues.sum())

    # 5. Length distribution (in characters)
    lengths = non_null_text.str.len()
    results["length_min"] = int(lengths.min())
    results["length_max"] = int(lengths.max())
    results["length_mean"] = round(float(lengths.mean()), 1)
    results["length_median"] = float(lengths.median())
    results["very_short_docs_under_20_chars"] = int((lengths < 20).sum())

    # 6. Category balance (informational, not a failure condition)
    results["category_counts"] = df["category"].value_counts().to_dict()

    return results


def write_report(results: dict, total_docs: int, path: str) -> None:
    lines = [
        "# Dataset Validation Summary — 20 Newsgroups\n",
        f"**Total documents loaded:** {total_docs}\n",
        "## Checks\n",
        f"- Null text entries: **{results['null_count']}**",
        f"- Empty / whitespace-only entries: **{results['empty_or_whitespace_count']}**",
        f"- Exact duplicate documents: **{results['duplicate_count']}**",
        f"- Documents with encoding issues: **{results['encoding_issue_count']}**",
        "",
        "## Document Length (characters)\n",
        f"- Min: {results['length_min']}",
        f"- Max: {results['length_max']}",
        f"- Mean: {results['length_mean']}",
        f"- Median: {results['length_median']}",
        f"- Suspiciously short (<20 chars): **{results['very_short_docs_under_20_chars']}**",
        "",
        "## Category Distribution\n",
    ]
    for category, count in sorted(results["category_counts"].items()):
        lines.append(f"- {category}: {count}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nValidation report written to {path}")


def main() -> int:
    df = acquire_dataset()

    df.to_csv(RAW_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Raw dataset saved to {RAW_OUTPUT_PATH}")

    results = validate_dataset(df)
    write_report(results, total_docs=len(df), path=VALIDATION_REPORT_PATH)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total documents:            {len(df)}")
    print(f"Null entries:                {results['null_count']}")
    print(f"Empty/whitespace entries:    {results['empty_or_whitespace_count']}")
    print(f"Duplicate documents:         {results['duplicate_count']}")
    print(f"Encoding issues:             {results['encoding_issue_count']}")
    print(f"Very short docs (<20 chars): {results['very_short_docs_under_20_chars']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
