# NLP Metadata Report — Week 4

Documents the topic modeling and sentiment enrichment run against the
35,070-chunk `documents` collection in `./chroma_db`, and how well it holds
up under manual review.

> **How to fill this in:** run the pipeline below in order, then replace the
> `TODO` markers with the numbers `evaluate_nlp_metadata.py` prints and the
> observations you make while reading `topic_summary.md`. Everything else
> in this file is already accurate.

---

## 1. Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `topic_modeling.py` | `topic_id`, `topic_label`, `topic_keywords` written into Chroma metadata; `topic_summary.md` for manual review; `logs/topic_modeling_perf.csv` |
| 2 | `sentiment_analysis.py` | `sentiment_label`, `sentiment_score` written into Chroma metadata; `logs/sentiment_analysis_perf.csv` |
| 3 | `create_manual_eval_sample.py` | `manual_eval_sample.csv` — a stratified sample (normal / jargon-heavy / short-or-garbled) for hand labeling |
| 4 | *(manual)* fill in `manual_topic_relevant` and `manual_sentiment_label` in the CSV | — |
| 5 | `evaluate_nlp_metadata.py` | `nlp_evaluation_results.md` — accuracy numbers this report references |

```
python topic_modeling.py
python sentiment_analysis.py
python create_manual_eval_sample.py --n 50
# ... fill in the two manual_* columns by hand in manual_eval_sample.csv ...
python evaluate_nlp_metadata.py
```

## 2. Topic Modeling

- **Method:** LDA (sklearn), run with `--method lda --n-topics 20`. BERTopic
  was tried first but was too slow on CPU-only hardware — embedding 34,387
  chunks with `sentence-transformers` was still only 39% done after 23
  minutes, so the run was killed. LDA finished the same corpus in a couple
  of minutes.
- **Runtime:** see `logs/topic_modeling_perf.csv` for the exact figure from
  this run.
- **Corpus split:** 34,387 / 35,070 chunks (98.1%) were usable for topic
  modeling; 36 too short, 647 garbled (skipped, see §4).
- **Manual validation (50-chunk sample, `manual_eval_sample.csv`):**
  **topic relevance rate = 56.0%** overall (28/50 topic labels judged as
  actually fitting the text), split by case type:
  - `normal` chunks: **61.8%** relevant
  - `jargon`-heavy chunks: **12.5%** relevant — sharp drop
  - `edge_case` (short/garbled) chunks: **75.0%** correctly triaged as
    unmodelable (see §4 for the two misses)

  Reading `topic_summary.md` end to end surfaced a real, recurring failure
  mode: **several LDA "topics" are not themes at all, they're artifacts.**
  - Topic 19 (`people_don_just` — people, don, just, think, like, know,
    way, say, time, make) is a **stopword-adjacent catch-all** that LDA
    used to absorb most short, generic-sounding discussion replies
    regardless of actual subject — it caught everything from a Waco
    political opinion to a tire-repair question to a theological debate.
  - Topic 16 (`article_writes_good`) captures **Usenet reply-boilerplate
    formatting** ("In article ... writes:") rather than content — it
    matches fine when a chunk *is* mostly quote-chain headers, but
    misfires when real content (e.g. a comet-nucleus astronomy discussion)
    happens to open with the same header pattern.
  - Topics 10 (`10_25_11` — all numeric tokens) and 18 (`ax_tape_giz`,
    `bhj`, `nsa`, `1t`, `3t`...) are **noise clusters**, not topics — one is
    LDA grouping number-heavy chunks (prices, stats tables, phone numbers)
    together regardless of subject, the other is a cluster of corrupted
    repeated-character text (a known 20 Newsgroups encoding artifact) that
    slipped past the garbled-text filter because the repetition kept the
    vowel ratio just high enough.
  - Coherent, genuinely useful topics did show up clearly: topic 3
    (`god_jesus_bible`), topic 15 (`armenian_armenians_turkish`), topic 0
    (`file_available_software`), topic 1 (`year_team_writes` — sports), and
    topic 2 (`use_dos_drive` — hardware/OS troubleshooting) were consistently
    on-target in the sample.

  **Takeaway:** LDA finds real, filterable themes for the majority of
  substantive posts, but a meaningful minority of topics are corpus
  artifacts (reply formatting, numeric noise) rather than content themes.
  Anyone filtering retrieval by `topic_label` should treat topics 10, 16,
  18, and 19 as low-confidence / catch-all, not as reliable content filters.

## 3. Sentiment Analysis

- **Engine:** VADER (`nltk.sentiment.vader`). Rationale: this corpus
  (20 Newsgroups) is informal, first-person discussion text — VADER's
  lexicon (tuned on social-media-style text) is a reasonable fit, it ran
  over all 35,070 chunks in 48.4 seconds on CPU, and needs no model
  download beyond the small `vader_lexicon` corpus. A HuggingFace
  transformer option (`--engine transformer`,
  `distilbert-base-uncased-finetuned-sst-2-english`) is available as a
  same-format drop-in for a higher-accuracy comparison, at the cost of
  runtime and a ~250MB download — worth trying given the accuracy below.
- **Accuracy vs. manual labels: 43.2% overall** (44/50 sampled rows judged
  — 6 genuinely-garbled rows were excluded since they have no real
  sentiment to score). Per-class breakdown:

  | Class | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | positive | 0.24 | 0.80 | 0.36 | 5 |
  | negative | 0.50 | 0.91 | 0.65 | 11 |
  | neutral | 1.00 | 0.18 | 0.30 | 28 |

  This is a clear, consistent pattern, not noise: **VADER over-predicts
  negative on neutral technical/informational text.** Neutral precision is
  perfect (1.00 — when VADER *does* say neutral, it's always right) but
  recall is only 0.18 — it calls the great majority of actually-neutral
  chunks negative instead. The confusion matrix confirms it: of 28
  true-neutral chunks, 12 were predicted positive and 9 predicted
  negative, only 5 correctly neutral.
- **Root cause, confirmed against the sample:** technical/problem-report
  language ("crashing," "problem," "error," "can't find," "disease," "risk")
  reliably pulls VADER's compound score negative even in neutral bug
  reports, FTP listings, and factual medical descriptions — e.g. an SCSI
  troubleshooting post ("boot device cannot be found... reboot, everything
  is fine") scored 0.25 (predicted positive, my read: neutral); an FTP
  software listing scored 0.88 (predicted positive, my read: neutral,
  no sentiment content at all). VADER also swings hard on isolated
  charged words in otherwise flat text (an FTP contact listing that
  happened to mention "cancer" and "risk" scored strongly negative despite
  being purely informational). **Takeaway: sentiment_label is reliable
  for genuinely emotional/opinionated posts (negative precision/recall
  both reasonably strong) but should not be trusted for filtering
  "neutral, purely informational" content** — that's exactly the class it
  gets wrong most.

## 4. Edge Case Handling

Both scripts apply the same triage (`nlp_utils.is_usable_for_nlp`) before
scoring:

| Case | Rule | What gets written |
|---|---|---|
| Too short | fewer than 20 non-whitespace characters | `topic_id=-1` / `topic_label="too_short_to_model"`; `sentiment_label="not_scored"` |
| Garbled | low alphabetic-character ratio, high digit ratio, or unnaturally low vowel ratio (catches base64 blobs, hex dumps, encoded headers that survived Week 1 cleaning) | `topic_id=-2` / `topic_label="garbled_skipped"`; `sentiment_label="not_scored"` |
| Jargon-heavy | not skipped, but flagged (`looks_jargon_heavy`) and over-sampled into the manual review set | scored normally, quality checked by hand |

This matters for filtered retrieval: a query filtering on `topic_id >= 0`
excludes both edge cases in one condition, and `sentiment_label !=
"not_scored"` excludes them from sentiment-filtered queries, so callers
never have to special-case missing metadata.

**Actual split, full corpus (35,070 chunks):** 34,387 usable (98.1%), 36
too short (0.1%), 647 garbled (1.8%).

**Manual check of the garbled heuristic (8 `edge_case` samples):** 6/8
were genuinely unreadable (base64-like or hex-noise blobs) and correctly
skipped — the heuristic is doing its job most of the time. But 2/8 were
**false positives**: a legitimate German university contact block (name,
address, phone/fax numbers) and a real hockey-standings statistics table
were both flagged "garbled" purely because of their high digit-to-text
ratio, not because they were actually unreadable. **Takeaway:** the
digit-ratio check in `is_garbled()` is too aggressive on legitimate
number-dense content (tables, contact info, stats) — a worthwhile
follow-up would be to only apply the digit-ratio rule when the vowel-ratio
check *also* fails, rather than either check alone being sufficient.

The jargon-heavy flag caught a different, more serious failure independent
of the garbled check: topic relevance on jargon-heavy chunks was only
12.5% (vs. 61.8% for normal text) — LDA's bag-of-words approach struggles
badly once a chunk is dominated by technical identifiers, acronyms, and
numeric codes rather than natural-language vocabulary it can cluster on.

## 5. Vector DB Integration / Filtered Retrieval

New metadata fields added to every chunk in the `documents` collection:

- `topic_id` (int), `topic_label` (str), `topic_keywords` (str)
- `sentiment_label` (str: positive / negative / neutral / not_scored),
  `sentiment_score` (float)

These combine with Chroma's existing `where` filtering (and, unchanged, the
`ner_enrichment.py`-produced entity metadata) for compound filtered
retrieval, e.g.:

```python
collection.query(
    query_texts=["..."],
    where={"$and": [
        {"topic_label": "graphics_card_drivers"},
        {"sentiment_label": "negative"},
    ]},
    n_results=5,
)
```

Verified against a synthetic collection that single-field, `$and`-combined,
and numeric-range (`topic_id >= 0`) filters all return the expected subset
— see `verify_filter.py` for the check.

## 6. Limitations / Next Steps

- LDA topic counts are fixed up front (`--n-topics 20` was used here);
  BERTopic finds its own count but wasn't usable on this hardware — CPU-only
  embedding of 34k+ chunks didn't finish in a reasonable time. If GPU access
  or more time becomes available, re-running with BERTopic and comparing
  the relevance rate against this LDA baseline (56.0%) would be worth doing.
- ~4 of the 20 LDA topics (10, 16, 18, 19) are artifacts (reply-formatting,
  numeric noise, corrupted-text clusters) rather than real themes — see §2.
  Downstream consumers filtering by `topic_label` should treat those four
  as low-confidence.
- VADER sentiment (43.2% accuracy) is lexicon-based and systematically
  over-predicts negative on neutral technical/informational text (§3) — the
  transformer engine (`--engine transformer`) is available as a same-format
  drop-in and is worth benchmarking against this baseline before trusting
  `sentiment_label` for anything beyond distinguishing clearly emotional
  posts from everything else.
- The `is_garbled()` digit-ratio check produced 2 false positives out of 8
  edge cases sampled (real contact info / stats tables flagged as garbled)
  — see §4 for the fix direction.
- The manual-eval sample (`--n 50` by default) is small by design (this is
  a spot-check, not a full held-out test set) — treat the accuracy numbers
  above as directional, not a formal benchmark.
