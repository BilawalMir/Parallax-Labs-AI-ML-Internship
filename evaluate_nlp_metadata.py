"""
evaluate_nlp_metadata.py
--------------------------
Week 4, deliverable 3 (part 2) + deliverable 5 input: reads the manually
filled-in manual_eval_sample.csv and computes:
  - Topic relevance rate (% of sampled chunks where the assigned topic
    label was judged relevant by a human reviewer)
  - Sentiment accuracy (predicted vs. manually-judged true label), plus
    per-class precision/recall/F1 and a confusion matrix
  - Same breakdown split out by case_type (normal / jargon / edge_case) so
    you can see where accuracy actually drops

Writes nlp_evaluation_results.md, which NLP_METADATA_REPORT.md references.

Usage:
    python evaluate_nlp_metadata.py --sample manual_eval_sample.csv
"""

import argparse
import csv
from collections import Counter, defaultdict


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(s):
    return (s or "").strip().lower()


def topic_relevance(rows):
    labeled = [r for r in rows if norm(r["manual_topic_relevant"]) in ("y", "n")]
    if not labeled:
        return None, 0
    relevant = sum(1 for r in labeled if norm(r["manual_topic_relevant"]) == "y")
    return relevant / len(labeled), len(labeled)


def sentiment_accuracy(rows):
    labeled = [r for r in rows if norm(r["manual_sentiment_label"]) in ("positive", "negative", "neutral")]
    if not labeled:
        return None, 0, None, {}
    correct = 0
    confusion = defaultdict(Counter)  # confusion[true][predicted]
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in labeled:
        true = norm(r["manual_sentiment_label"])
        pred = norm(r["predicted_sentiment_label"])
        confusion[true][pred] += 1
        if pred == true:
            correct += 1
            per_class[true]["tp"] += 1
        else:
            per_class[true]["fn"] += 1
            per_class[pred]["fp"] += 1

    metrics = {}
    for label in ("positive", "negative", "neutral"):
        tp, fp, fn = per_class[label]["tp"], per_class[label]["fp"], per_class[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return correct / len(labeled), len(labeled), confusion, metrics


def by_case_type(rows, fn):
    out = {}
    for case_type in ("normal", "jargon", "edge_case"):
        subset = [r for r in rows if r.get("case_type") == case_type]
        if subset:
            out[case_type] = fn(subset)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="manual_eval_sample.csv")
    parser.add_argument("--out", default="nlp_evaluation_results.md")
    args = parser.parse_args()

    rows = load_rows(args.sample)

    rel_rate, n_topic_labeled = topic_relevance(rows)
    sent_acc, n_sent_labeled, confusion, per_class = sentiment_accuracy(rows)

    lines = ["# NLP Metadata Evaluation Results\n"]
    lines.append(f"Sample size: {len(rows)} chunks from `{args.sample}`.\n")

    lines.append("## Topic Relevance")
    if rel_rate is None:
        lines.append("_No rows had `manual_topic_relevant` filled in yet._\n")
    else:
        lines.append(f"- Relevance rate: **{rel_rate*100:.1f}%** ({n_topic_labeled} rows judged)\n")
        by_case = by_case_type(rows, lambda subset: topic_relevance(subset)[0])
        for case_type, rate in by_case.items():
            if rate is not None:
                lines.append(f"  - `{case_type}`: {rate*100:.1f}%")
        lines.append("")

    lines.append("## Sentiment Accuracy")
    if sent_acc is None:
        lines.append("_No rows had `manual_sentiment_label` filled in yet._\n")
    else:
        lines.append(f"- Overall accuracy: **{sent_acc*100:.1f}%** ({n_sent_labeled} rows judged)\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for label, m in per_class.items():
            lines.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
        lines.append("")

        lines.append("**Confusion matrix (rows = true label, cols = predicted):**\n")
        labels = ["positive", "negative", "neutral"]
        header = "| true \\ pred | " + " | ".join(labels) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(labels) + 1))
        for true_label in labels:
            row_counts = [str(confusion[true_label].get(p, 0)) for p in labels]
            lines.append(f"| {true_label} | " + " | ".join(row_counts) + " |")
        lines.append("")

        by_case = by_case_type(rows, lambda subset: sentiment_accuracy(subset)[0])
        lines.append("**Accuracy by case type:**")
        for case_type, acc in by_case.items():
            if acc is not None:
                lines.append(f"- `{case_type}`: {acc*100:.1f}%")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")
    if rel_rate is not None:
        print(f"Topic relevance: {rel_rate*100:.1f}%")
    if sent_acc is not None:
        print(f"Sentiment accuracy: {sent_acc*100:.1f}%")


if __name__ == "__main__":
    main()
