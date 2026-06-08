# Implementation Plan — Tecator Fat Regression

Derived from `2026-06-08-tecator-fat-regression-design.md`. Executed subagent-driven.

## Wave 1 — Independent modules (parallel subagents)
Each agent writes the module + pytest tests and runs them green before returning.

- **T1 `tecator/data.py`** — `WAVELENGTHS`, `load_data`, `make_holdout` (stratified by fat
  quantile bins), `repeated_cv`. Tests: shapes, no NaN, determinism, holdout size 43, test set
  spans fat range.
- **T2 `tecator/preprocessing.py`** — `StandardNormalVariate`, `MultiplicativeScatterCorrection`,
  `SavitzkyGolay(window_length, polyorder, deriv)`. Tests: SNV row stats, SavGol shape & deriv,
  sklearn-clonable, pipeline-safe.
- **T3 `tecator/evaluation.py`** — metrics (`rmse`,`mae`,`r2`,`sep`), `cross_validate_model`,
  `evaluate_on_test`, plot helpers saving to `results/figures/`. Tests: metrics on toy data.

## Wave 2 — Models (depends on T1–T3)
- **T4 `tecator/models.py`** — `build_models`, `param_grids`, `pls_component_scan`. Smoke test:
  every pipeline fits + predicts on a small slice without error.

## Wave 3 — Experiments (depends on T4) — run by orchestrator
- **T5 `run_experiments.py`** — full pipeline → `results/cv_metrics.csv`,
  `results/test_metrics.json`, `results/figures/*.png`, `results/summary.md`. Verify metrics hit
  success criteria (R² > 0.98, RMSE well below 2.6).

## Wave 4 — Narrative + docs (depends on T5, parallel)
- **T6 `tecator_fat_regression.ipynb`** — narrative notebook executed end-to-end (real outputs).
- **T7 `README.md`** — visually appealing: badges, problem, method, results table, embedded
  figures, leakage note, how-to-run. Plus `requirements.txt`.

## Wave 5 — Ship
- **T8** — final full test run, commit, push to `main`.

## Shared conventions (all agents)
- Use `.venv/bin/python` for running anything.
- Spectra matrix `X` is `(n, 100)` float64; target `y` is fat % float64.
- All preprocessing must be sklearn transformers usable inside `Pipeline` (no global fit).
- random_state=42 everywhere for reproducibility.
- Keep deps to: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, pytest.
