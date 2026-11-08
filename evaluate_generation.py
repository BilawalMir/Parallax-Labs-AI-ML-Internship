"""
evaluate_generation.py
-----------------------
Automated end-to-end GENERATION evaluation for the FastAPI RAG service
(Task 5). Companion script to evaluate_retrieval.py, same pattern: hits the
LIVE running API over HTTP and writes JSON + Markdown reports to
eval_reports/.

What this measures
-------------------
Unlike retrieval (precision/recall against a category label), there's no
hand-labeled "correct answer" key for generation quality here. So this
script reuses the grounding signal that already exists in the pipeline:
generate_answer.py's check_grounding() computes a lexical-overlap heuristic
between the generated answer and the retrieved context, exposed as
QueryResponse.grounded. "Grounded rate" (fraction of ANSWERED responses
with grounded=true) is the quality metric here -- a proxy for "did the
model stick to the provided context" rather than a factuality judgment.

Also reported:
  - status distribution across the four GenerationResult buckets:
    answered / abstained / out_of_domain / error
  - latency_s per query (retrieval, generation, total) + aggregate mean/
    p50/p95, so you can see how much of total latency is the LLM call vs.
    retrieval

Why the test set includes a few out-of-domain queries
-------------------------------------------------------
If every test query were comfortably in-domain, the status distribution
would only ever show "answered" (or "error" if the API key/network is
down), and the out_of_domain gate (RAGPipeline's retrieval-confidence
threshold) would never actually get exercised. So on top of the 12
in-domain queries (same ones evaluate_retrieval.py uses, for
cross-script comparability), this test set adds a handful of queries with
no business being answered from a 20-Newsgroups-derived corpus, to check
that the pipeline abstains/gates instead of hallucinating.

Requires: the FastAPI service running (either backend -- generation doesn't
need topic/sentiment metadata, so unlike evaluate_retrieval.py this does
NOT require the Chroma backend). Also requires OPENROUTER_API_KEY to be set
for "answered" results -- if it's missing, every in-domain query will
legitimately come back with status="error", and that's still a valid (if
uninteresting) eval run, not a script bug.

Usage:
    python -m uvicorn api:app --port 8000    # in one terminal (PowerShell)
    python evaluate_generation.py             # in another
    python evaluate_generation.py --api-url http://127.0.0.1:8000 --timeout 120
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from statistics import mean

import requests

REPORT_DIR = "eval_reports"

# --------------------------------------------------------------------------
# Test set. Reuses the 12 in-domain queries from evaluate_retrieval.py
# (same wording, so retrieval and generation reports line up query-for-
# query) plus a few queries chosen to be clearly outside the 20-Newsgroups
# corpus, to exercise the out_of_domain gate.
# --------------------------------------------------------------------------
TEST_SET = [
    {"question": "how do spacecraft communicate with mission control over long distances", "expected_domain": "in_domain"},
    {"question": "what causes epileptic seizures and how are they treated", "expected_domain": "in_domain"},
    {"question": "how does public key encryption keep data secure", "expected_domain": "in_domain"},
    {"question": "best practices for maintaining a car engine", "expected_domain": "in_domain"},
    {"question": "tips for riding a motorcycle safely in traffic", "expected_domain": "in_domain"},
    {"question": "who won the hockey championship this season", "expected_domain": "in_domain"},
    {"question": "how to render 3D graphics with a GPU", "expected_domain": "in_domain"},
    {"question": "troubleshooting a Windows operating system crash", "expected_domain": "in_domain"},
    {"question": "the meaning of faith and prayer in Christianity", "expected_domain": "in_domain"},
    {"question": "arguments for and against stricter gun control laws", "expected_domain": "in_domain"},
    {"question": "items for sale: used electronics and computer parts", "expected_domain": "in_domain"},
    {"question": "debate about the existence of God from an atheist perspective", "expected_domain": "in_domain"},
    {"question": "what's the best recipe for chocolate chip cookies", "expected_domain": "out_of_domain"},
    {"question": "recommend a good romantic comedy movie to watch tonight", "expected_domain": "out_of_domain"},
    {"question": "how do I train for a marathon as a beginner runner", "expected_domain": "out_of_domain"},
]

# "expected_domain" is informational only (shown in the report so you can
# eyeball whether the gate did what you'd expect) -- it is NOT asserted as
# pass/fail. Retrieval confidence on unfamiliar wording is fuzzy enough
# that treating a mismatch as a hard failure would produce noisy, unfair
# "failures" rather than a useful signal.


# --------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------

def query_api(api_base_url: str, question: str, timeout: int) -> dict:
    resp = requests.post(
        f"{api_base_url}/api/v1/query",
        json={"question": question},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"/api/v1/query returned {resp.status_code} for question {question!r}: {resp.text}"
        )
    return resp.json()


def check_service_ready(api_base_url: str) -> None:
    """Unlike evaluate_retrieval.py, generation works on either retriever
    backend (Chroma or TF-IDF fallback), so this only confirms the service
    is up -- no backend restriction."""
    try:
        resp = requests.get(f"{api_base_url}/health", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Could not reach {api_base_url}. Is the service running?\n"
            f"  python -m uvicorn api:app --port 8000"
        )
    backend = resp.json()["retriever_backend"]
    print(f"Service is up (retriever_backend={backend}).")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (0-100), no numpy dependency.
    Matches the common 'inclusive' method. Returns 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[int(rank)]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p95": round(percentile(values, 95), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------

def _clip_for_table(text: str, max_len: int = 120) -> str:
    """Markdown table cells can't contain raw newlines or unescaped pipes."""
    flat = text.replace("\n", " ").replace("|", "\\|").strip()
    if len(flat) > max_len:
        flat = flat[: max_len - 1].rstrip() + "\u2026"
    return flat


def write_reports(per_query_rows: list[dict], summary: dict, api_base_url: str) -> tuple[str, str]:
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(REPORT_DIR, f"generation_eval_{ts}.json")
    md_path = os.path.join(REPORT_DIR, f"generation_eval_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "api_base_url": api_base_url,
                "num_queries": len(per_query_rows),
                "summary": summary,
                "per_query": per_query_rows,
            },
            f,
            indent=2,
        )

    lines = [
        f"# Generation Evaluation Report ({ts})",
        "",
        f"- Service: `{api_base_url}`",
        f"- Test queries: {len(per_query_rows)}",
        "",
        "## Status distribution",
        "",
        "| Status | Count | % of total |",
        "|---|---|---|",
    ]
    total = len(per_query_rows)
    for status, count in summary["status_counts"].items():
        pct = (count / total * 100) if total else 0.0
        lines.append(f"| {status} | {count} | {pct:.1f}% |")

    lines += [
        "",
        f"## Grounding (among {summary['num_answered']} 'answered' responses)",
        "",
        f"- Grounded: {summary['num_grounded']} / {summary['num_answered']} "
        f"({summary['grounded_rate'] * 100:.1f}%)" if summary["num_answered"] else
        "- No 'answered' responses in this run -- grounding rate not applicable.",
        "",
        "## Latency (seconds, across all queries)",
        "",
        "| Metric | Mean | p50 | p95 | Min | Max |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in [("Total", "total_latency_s"), ("Retrieval", "retrieval_latency_s"), ("Generation", "generation_latency_s")]:
        st = summary["latency"][key]
        lines.append(f"| {label} | {st['mean']:.3f} | {st['p50']:.3f} | {st['p95']:.3f} | {st['min']:.3f} | {st['max']:.3f} |")

    lines += [
        "",
        "## Per-query results",
        "",
        "| Question | Expected domain | Status | Grounded | Total (s) | Answer (truncated) |",
        "|---|---|---|---|---|---|",
    ]
    for row in per_query_rows:
        grounded_disp = "-" if row["grounded"] is None else ("yes" if row["grounded"] else "no")
        lines.append(
            f"| {row['question']} | {row['expected_domain']} | {row['status']} | "
            f"{grounded_disp} | {row['total_latency_s']:.2f} | {_clip_for_table(row['answer'])} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_evaluation(api_base_url: str, timeout: int) -> None:
    check_service_ready(api_base_url)

    per_query_rows = []
    status_counts: dict[str, int] = {}

    print(f"\nRunning {len(TEST_SET)} test queries against {api_base_url}/api/v1/query ...")
    for item in TEST_SET:
        question, expected_domain = item["question"], item["expected_domain"]
        try:
            data = query_api(api_base_url, question, timeout)
        except (requests.exceptions.Timeout, RuntimeError) as e:
            print(f"  {question!r} -> FAILED: {e}")
            row = {
                "question": question,
                "expected_domain": expected_domain,
                "status": "script_error",
                "grounded": None,
                "answer": str(e),
                "model": None,
                "retrieval_latency_s": 0.0,
                "generation_latency_s": 0.0,
                "total_latency_s": 0.0,
                "num_chunks_retrieved": 0,
            }
            per_query_rows.append(row)
            status_counts["script_error"] = status_counts.get("script_error", 0) + 1
            continue

        row = {
            "question": question,
            "expected_domain": expected_domain,
            "status": data["status"],
            "grounded": data["grounded"],
            "answer": data["answer"],
            "model": data["model"],
            "retrieval_latency_s": data["retrieval_latency_s"],
            "generation_latency_s": data["generation_latency_s"],
            "total_latency_s": data["total_latency_s"],
            "num_chunks_retrieved": len(data["chunks"]),
        }
        per_query_rows.append(row)
        status_counts[data["status"]] = status_counts.get(data["status"], 0) + 1

        grounded_disp = "-" if data["grounded"] is None else data["grounded"]
        print(
            f"  {question!r} -> status={data['status']} grounded={grounded_disp} "
            f"total={data['total_latency_s']:.2f}s"
        )

    answered_rows = [r for r in per_query_rows if r["status"] == "answered"]
    num_grounded = sum(1 for r in answered_rows if r["grounded"])
    grounded_rate = (num_grounded / len(answered_rows)) if answered_rows else 0.0

    summary = {
        "status_counts": status_counts,
        "num_answered": len(answered_rows),
        "num_grounded": num_grounded,
        "grounded_rate": round(grounded_rate, 4),
        "latency": {
            "total_latency_s": latency_stats([r["total_latency_s"] for r in per_query_rows]),
            "retrieval_latency_s": latency_stats([r["retrieval_latency_s"] for r in per_query_rows]),
            "generation_latency_s": latency_stats([r["generation_latency_s"] for r in per_query_rows]),
        },
    }

    print("\n=== Summary ===")
    print(f"  Status distribution: {status_counts}")
    if answered_rows:
        print(f"  Grounded rate (of 'answered'): {num_grounded}/{len(answered_rows)} ({grounded_rate * 100:.1f}%)")
    else:
        print("  No 'answered' responses -- grounded rate not applicable.")
    lat = summary["latency"]["total_latency_s"]
    print(f"  Total latency: mean={lat['mean']:.3f}s  p50={lat['p50']:.3f}s  p95={lat['p95']:.3f}s")

    json_path, md_path = write_reports(per_query_rows, summary, api_base_url)
    print(f"\nReports written:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG API's generation quality and latency.")
    parser.add_argument("--api-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request HTTP timeout in seconds (default 120 -- generation can take a while: "
             "OpenRouter calls retry up to 3x with backoff inside the pipeline before giving up).",
    )
    args = parser.parse_args()
    run_evaluation(args.api_url, args.timeout)


if __name__ == "__main__":
    main()
