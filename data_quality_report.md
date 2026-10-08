# Data Quality Report — Post-Cleaning

**Total documents processed:** 18846

## Cleaning Impact

- Average original length: **1902.5 characters**
- Average cleaned length: **1162.0 characters**
- Average size reduction: **38.9%** (headers, quotes, signatures, HTML, URLs, emails, emoji removed)
- Documents that became empty after cleaning (pure noise/headers): **52**
- Very short cleaned documents (<20 chars): **36**
- Duplicate documents after cleaning (not caught by raw-text duplicate check): **119**

## Language Distribution (top 10, post-cleaning)

- en: 18709 (99.27%)
- unknown: 53 (0.28%)
- de: 17 (0.09%)
- fr: 9 (0.05%)
- ro: 9 (0.05%)
- tl: 7 (0.04%)
- af: 6 (0.03%)
- ca: 6 (0.03%)
- nl: 5 (0.03%)
- et: 5 (0.03%)

## Notes

- `language == 'unknown'` means the cleaned text was empty or too short/ambiguous for reliable detection — expected for near-empty posts after header/quote removal.
- Documents that became empty after cleaning are still present in `processed_dataset.csv` (not silently dropped) so downstream steps can decide how to handle them — flagging this as a decision point rather than assuming they should be deleted.