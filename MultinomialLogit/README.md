# MultinomialLogit

This folder contains the multinomial logit (LASSO) pipeline and its outputs.

## Structure

- `run_sector_models_smoke50.py`
  - Main script for sector-level training and evaluation.
- `sp500_panel_multinomial_logit_lasso.ipynb`
  - Explanatory notebook for the modeling workflow.
- `date_coverage_audit.csv`
  - Date coverage audit for quality control.
- `runs/`
  - Organized output folders for model runs.

## `runs/` folder

- `runs/legacy/`
  - Historical runs that are versioned for reference.
- `runs/local/` (git-ignored)
  - New local runs (smoke/full) for day-to-day work.
  - This is where current script outputs are written.

## Run naming convention

- `smoke_test_<N>_sector_models_<timestamp>`
- `sector_models_<N>_<timestamp>`

## Expected outputs per run

- metrics (`all_sector_metrics_long.csv`, `all_sector_metrics_wide.csv`)
- validation predictions (`all_sector_validation_predictions.csv`)
- hyperparameter summaries and coefficients
- sector-level plots and reports

## Repository hygiene

- IDE artifacts (`.jupyter`, `.ipynb_checkpoints`) are not versioned.
- Local run outputs are not versioned; only legacy/final selected runs are tracked.
