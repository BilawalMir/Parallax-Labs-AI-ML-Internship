# NLP Metadata Evaluation Results

Sample size: 50 chunks from `manual_eval_sample.csv`.

## Topic Relevance
- Relevance rate: **56.0%** (50 rows judged)

  - `normal`: 61.8%
  - `jargon`: 12.5%
  - `edge_case`: 75.0%

## Sentiment Accuracy
- Overall accuracy: **43.2%** (44 rows judged)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.24 | 0.80 | 0.36 | 5 |
| negative | 0.50 | 0.91 | 0.65 | 11 |
| neutral | 1.00 | 0.18 | 0.30 | 28 |

**Confusion matrix (rows = true label, cols = predicted):**

| true \ pred | positive | negative | neutral |
|---|---|---|---|
| positive | 4 | 1 | 0 |
| negative | 1 | 10 | 0 |
| neutral | 12 | 9 | 5 |

**Accuracy by case type:**
- `normal`: 47.1%
- `jargon`: 37.5%
- `edge_case`: 0.0%
