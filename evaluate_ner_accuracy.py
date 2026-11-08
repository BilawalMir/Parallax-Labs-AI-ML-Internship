"""
evaluate_ner_accuracy.py
--------------------------
Runs spaCy's NER on gold_standard_ner.py's manually-labeled excerpts and
reports precision/recall/F1 -- the accuracy evaluation required by the
task ("evaluating accuracy against a small manually labeled set").

Reports two scores, because they answer different questions:
  - Exact match (entity text + label both correct): the strict number --
    "did spaCy get this exactly right for filtered retrieval by type?"
  - Span-only match (entity text correct, label ignored): looser --
    "did spaCy at least notice this was a named thing?" Useful because a
    span-correct/label-wrong error (e.g. "Koresh" tagged WORK_OF_ART
    instead of PERSON) is a different, less severe failure than missing
    the entity outright, and both matter for understanding metadata
    quality differently.

Usage:
    python evaluate_ner_accuracy.py
"""

from __future__ import annotations

from gold_standard_ner import GOLD_STANDARD

RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "NORP"}


def normalize(text: str) -> str:
    return text.strip().lower()


def get_predicted_entities(text: str, nlp) -> set[tuple[str, str]]:
    """Runs spaCy on `text`, returns (normalized_text, label) pairs
    restricted to the four label types this project cares about -- an
    entity spaCy tags as e.g. DATE or CARDINAL was never a candidate for
    the gold standard in the first place, so counting it as a false
    positive here would penalize the model for something out of scope."""
    doc = nlp(text)
    return {
        (normalize(ent.text), ent.label_)
        for ent in doc.ents
        if ent.label_ in RELEVANT_LABELS
    }


def score_example(gold_entities: list[tuple[str, str]], predicted: set[tuple[str, str]]) -> dict:
    gold_exact = {(normalize(t), label) for t, label in gold_entities}
    gold_span = {normalize(t) for t, _ in gold_entities}
    predicted_span = {t for t, _ in predicted}

    return {
        "exact_tp": len(gold_exact & predicted),
        "exact_fp": len(predicted - gold_exact),
        "exact_fn": len(gold_exact - predicted),
        "span_tp": len(gold_span & predicted_span),
        "span_fp": len(predicted_span - gold_span),
        "span_fn": len(gold_span - predicted_span),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(model_name: str = "en_core_web_sm") -> dict:
    import spacy

    nlp = spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    totals = {"exact_tp": 0, "exact_fp": 0, "exact_fn": 0, "span_tp": 0, "span_fp": 0, "span_fn": 0}
    per_example_errors = []

    for example in GOLD_STANDARD:
        predicted = get_predicted_entities(example["text"], nlp)
        scores = score_example(example["entities"], predicted)
        for key in totals:
            totals[key] += scores[key]

        gold_exact = {(normalize(t), label) for t, label in example["entities"]}
        missed = gold_exact - predicted
        extra = predicted - gold_exact
        if missed or extra:
            per_example_errors.append({"text": example["text"], "missed": missed, "extra": extra})

    exact_p, exact_r, exact_f1 = precision_recall_f1(totals["exact_tp"], totals["exact_fp"], totals["exact_fn"])
    span_p, span_r, span_f1 = precision_recall_f1(totals["span_tp"], totals["span_fp"], totals["span_fn"])

    print(f"Evaluated on {len(GOLD_STANDARD)} manually-labeled excerpts "
          f"({sum(len(e['entities']) for e in GOLD_STANDARD)} gold entities).\n")

    print("Exact match (text + label):")
    print(f"  Precision: {exact_p:.3f}   Recall: {exact_r:.3f}   F1: {exact_f1:.3f}")
    print(f"  TP={totals['exact_tp']} FP={totals['exact_fp']} FN={totals['exact_fn']}\n")

    print("Span-only match (text, label ignored):")
    print(f"  Precision: {span_p:.3f}   Recall: {span_r:.3f}   F1: {span_f1:.3f}")
    print(f"  TP={totals['span_tp']} FP={totals['span_fp']} FN={totals['span_fn']}\n")

    print(f"Examples with at least one mismatch: {len(per_example_errors)}/{len(GOLD_STANDARD)}")
    for err in per_example_errors:
        print(f"\n  Text: {err['text'][:90]}...")
        if err["missed"]:
            print(f"    Missed (in gold, not predicted): {err['missed']}")
        if err["extra"]:
            print(f"    Extra (predicted, not in gold):  {err['extra']}")

    return {
        "exact_precision": exact_p, "exact_recall": exact_r, "exact_f1": exact_f1,
        "span_precision": span_p, "span_recall": span_r, "span_f1": span_f1,
    }


if __name__ == "__main__":
    evaluate()
