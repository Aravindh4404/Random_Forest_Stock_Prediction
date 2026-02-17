# ConstructionDataset

This folder contains the construction pipeline for the final ticker-day dataset.

## Structure

- `claude_lagsentiment_dividends_btm.py`
  - Main script to build features per ticker.
- `feature_datasets_enhanced/`
  - Final ticker-level dataset (`*_features.csv`), one file per company.
  - This is the **source of truth** for modeling.
- `data_dictionary/`
  - Data dictionary and variable descriptions.
- `_processing_summary_latest.csv`
  - Summary of the latest dataset build process.
- `cache/` (git-ignored)
  - Local consolidated-panel cache to speed up model runs.
- `smoke_tests/` (git-ignored)
  - Functional smoke tests for dataset construction.
- `all_companies_features.csv` (git-ignored)
  - Large local consolidated panel (not versioned due to size).

## Recommended flow

1. Build/update ticker-level features with `claude_lagsentiment_dividends_btm.py`.
2. Validate quality with smoke tests in `smoke_tests/`.
3. Train models from ticker-level files (not from one giant CSV).
4. Build a consolidated panel only at runtime when a model needs it.

## Notes

- Heavy artifacts (`all_companies_features.csv`, `cache/`) are excluded from git.
- If you need a consolidated panel, generate it at runtime instead of using it as the primary source.
