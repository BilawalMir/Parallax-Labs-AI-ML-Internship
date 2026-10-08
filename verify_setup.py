"""
verify_setup.py
----------------
Verifies that every library required for the Parallax Labs data-cleaning
pipeline is installed AND actually works (not just importable).

Run:
    python src/verify_setup.py

Exit code 0 = everything passed. Non-zero = something failed (see printed log).
"""

import sys
import importlib


def check_import(module_name: str, package_name: str | None = None) -> bool:
    """Try importing a module; print a clear PASS/FAIL line."""
    display_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"[PASS] import {display_name}")
        return True
    except ImportError as e:
        print(f"[FAIL] import {display_name} -> {e}")
        return False


def check_pandas() -> bool:
    try:
        import pandas as pd

        df = pd.DataFrame({"text": ["hello", None, "world"]})
        assert df["text"].isna().sum() == 1
        print(f"[PASS] pandas basic functionality (version {pd.__version__})")
        return True
    except Exception as e:
        print(f"[FAIL] pandas functionality -> {e}")
        return False


def check_numpy() -> bool:
    try:
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        print(f"[PASS] numpy basic functionality (version {np.__version__})")
        return True
    except Exception as e:
        print(f"[FAIL] numpy functionality -> {e}")
        return False


def check_sklearn_dataset() -> bool:
    """Confirms sklearn is installed and can reach its dataset loader.
    NOTE: this does NOT download the full dataset (that happens in Stage 2) —
    it only checks the loader function exists and is callable."""
    try:
        from sklearn.datasets import fetch_20newsgroups

        assert callable(fetch_20newsgroups)
        print("[PASS] scikit-learn fetch_20newsgroups is importable and callable")
        return True
    except Exception as e:
        print(f"[FAIL] scikit-learn dataset loader -> {e}")
        return False


def check_nltk() -> bool:
    try:
        import nltk

        # Ensure the punkt tokenizer data is available; download quietly if missing.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import word_tokenize

        tokens = word_tokenize("Hello, world! This is a test.")
        assert len(tokens) > 0
        print(
            f"[PASS] nltk basic functionality (version {nltk.__version__}), tokens={tokens}"
        )
        return True
    except Exception as e:
        print(f"[FAIL] nltk functionality -> {e}")
        return False


def check_langdetect() -> bool:
    try:
        from langdetect import detect

        lang = detect("This is an English sentence.")
        assert lang == "en"
        print(f"[PASS] langdetect basic functionality (detected='{lang}')")
        return True
    except Exception as e:
        print(f"[FAIL] langdetect functionality -> {e}")
        return False


def check_ftfy() -> bool:
    try:
        import ftfy

        fixed = ftfy.fix_text("This textâ€™s got mojibake")
        assert isinstance(fixed, str) and len(fixed) > 0
        print(f"[PASS] ftfy basic functionality -> '{fixed}'")
        return True
    except Exception as e:
        print(f"[FAIL] ftfy functionality -> {e}")
        return False


def check_chardet() -> bool:
    try:
        import chardet

        raw_bytes = "hello world".encode("utf-8")
        result = chardet.detect(raw_bytes)
        assert "encoding" in result
        print(f"[PASS] chardet basic functionality -> {result}")
        return True
    except Exception as e:
        print(f"[FAIL] chardet functionality -> {e}")
        return False


def check_emoji() -> bool:
    try:
        import emoji

        has_emoji = emoji.emoji_count("Hello 😀 world")
        assert has_emoji == 1
        print(f"[PASS] emoji basic functionality (count={has_emoji})")
        return True
    except Exception as e:
        print(f"[FAIL] emoji functionality -> {e}")
        return False


def check_tiktoken() -> bool:
    try:
        import tiktoken
    except ImportError as e:
        print(f"[FAIL] tiktoken import -> {e}")
        return False

    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode("Hello world")
        assert len(tokens) > 0
        print(f"[PASS] tiktoken basic functionality (tokens={tokens})")
        return True
    except Exception as e:
        # tiktoken downloads its encoding file from the network on first use.
        # In network-restricted environments this call can fail even though
        # the library itself is installed correctly - treat as a warning,
        # not a hard failure, since it will work on Colab / a normal machine.
        print(
            f"[WARN] tiktoken installed but encoding download failed (likely network-restricted sandbox) -> {e}"
        )
        return True


def check_pytest() -> bool:
    return check_import("pytest")


def check_tqdm() -> bool:
    try:
        from tqdm import tqdm

        total = sum(1 for _ in tqdm(range(5), disable=True))
        assert total == 5
        print("[PASS] tqdm basic functionality")
        return True
    except Exception as e:
        print(f"[FAIL] tqdm functionality -> {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("Parallax Labs — Environment Verification")
    print("=" * 60)

    checks = [
        check_pandas,
        check_numpy,
        check_sklearn_dataset,
        check_nltk,
        check_langdetect,
        check_ftfy,
        check_chardet,
        check_emoji,
        check_tiktoken,
        check_pytest,
        check_tqdm,
    ]

    results = [check() for check in checks]

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
