# Tecator Fat-Content Regression — Design

**Date:** 2026-06-08
**Status:** Approved
**Author:** Ritvik Maini (with Claude Code)

## 1. Problem & Motivation

The Tecator benchmark (OpenML dataset 505) asks us to **predict the fat content (%) of a
meat sample from its 100-channel near-infrared (NIR) absorbance spectrum** (850–1050 nm).
The default target attribute is `fat` and the task is **regression**.

The existing repo (`challenge2.ipynb`) instead reframes the problem as binary classification
(`fat > 20%`) and reports up to **100% test accuracy**. That number is not trustworthy:

1. **Data leakage** — `StandardScaler` is fit on the *entire* dataset *before* the train/test
   split, so the scaler sees the test data.
2. **Tiny single split** — 43 test samples in one split; 100% is not a robust estimate.
3. **Wrong task** — the canonical Tecator task is regression on `fat`, not a thresholded class.

This project replaces that with a rigorous, leak-free regression solution that uses the
domain-correct chemometric methods and reports trustworthy, reproducible metrics.

## 2. Goals & Success Criteria

- Predict `fat` (%) from the NIR spectrum (regression).
- **Zero leakage**: every preprocessing step lives inside an sklearn `Pipeline`, fit only on
  training folds.
- **Trustworthy evaluation**: repeated stratified K-fold CV on the training set for model
  selection; a held-out test set touched exactly once.
- **Metrics**: RMSE, MAE, R², and SEP (Standard Error of Prediction), reported as mean ± std.
- **Target performance**: R² > 0.98 and RMSE meaningfully below the ~2.6 PLS-on-raw baseline
  (aim ≈1 with 2nd-derivative preprocessing).
- **Deliverables**: reusable `tecator/` package, a narrative notebook, a visually appealing
  README with embedded figures and results, all committed and pushed to `main`.

## 3. Background (research-grounded)

- Spectral **derivatives** (1st/2nd, via Savitzky-Golay) are consistently more informative than
  raw absorbances on Tecator — the classic finding (Borggaard & Thodberg, 1992).
- **PLS regression** is the chemometric gold standard for NIR (PLS1 ≈ RMSE 2.6 on raw spectra;
  much lower with derivative preprocessing).
- Standard NIR scatter-correction transforms: **SNV** (Standard Normal Variate) and
  **MSC** (Multiplicative Scatter Correction).

## 4. Architecture

New `tecator/` Python package (sklearn-compatible, lightweight — numpy/pandas/scikit-learn/
scipy/matplotlib/seaborn; **no** TensorFlow/Keras).

### `tecator/data.py`
- `WAVELENGTHS` — np.ndarray of the 100 channel wavelengths.
- `load_data(path="data/tecator.csv") -> (X, targets, wavelengths)` where `X` is `(n, 100)`
  float spectra and `targets` is a DataFrame with `fat`, `water`, `protein`.
- `make_holdout(X, y, test_size=43, random_state=42) -> (X_tr, X_te, y_tr, y_te)` —
  reproducible split, **stratified by fat quantile bins** so the test set spans the fat range.
- `repeated_cv(n_splits=5, n_repeats=10, random_state=42)` — returns a CV splitter
  (RepeatedKFold) for use across all model comparisons.

### `tecator/preprocessing.py`
sklearn `BaseEstimator, TransformerMixin` transformers, each `(n,100) -> (n,100)`:
- `StandardNormalVariate` — row-wise center & scale each spectrum.
- `MultiplicativeScatterCorrection` — fit a reference mean spectrum, correct each row by OLS.
- `SavitzkyGolay(window_length=11, polyorder=2, deriv=0)` — `scipy.signal.savgol_filter`
  along the wavelength axis; `deriv=1/2` produce derivative spectra.

### `tecator/models.py`
- `build_models(random_state=42) -> dict[str, Pipeline]` — each value is a full `Pipeline`
  (preprocessing → scaler → estimator) so there is no leakage. Includes:
  - `PLS` (raw), `PLS + SNV`, `PLS + 2nd derivative`
  - `PCR` (PCA → linear), `Ridge`, `SVR (RBF)`, `GradientBoosting`, `MLP` (sklearn MLPRegressor)
  - **Best-in-class additions (approved 2026-06-08):**
    - `KernelRidge (2nd deriv)` — RBF KernelRidge on 2nd-derivative spectra (documented top
      performer for Tecator).
    - `GaussianProcess (2nd deriv)` — GPR with RBF + WhiteKernel on 2nd-derivative spectra.
    - `Stacking` — `StackingRegressor` fusing PLS(2nd-deriv) + SVR + KernelRidge (+ GPR) with a
      Ridge meta-learner; the "fuse complementary models" approach.
- `param_grids() -> dict[str, dict]` — small, sane hyperparameter grids for tuning
  (e.g. PLS `n_components`, SVR `C`/`gamma`, Ridge `alpha`, KernelRidge `alpha`/`gamma`).
- `pls_component_scan(X, y, cv, max_components, preprocessor)` — RMSE vs n_components curve.
- `vip_scores(pls_pipeline, X, y) -> np.ndarray` — Variable Importance in Projection per
  wavelength (PLS-based diagnostic for the README/notebook).

**Performance bar:** best-in-class on this benchmark is RMSE < 1.0 / R² > 0.99 (PLS-on-raw ≈ 2.61;
neural net ≈ 0.85; kernel-ridge/GP + 2nd-derivative among the best published). The kernel models
and stacking ensemble are what move us from "good" (~RMSE 1–1.5) into that range.

### `tecator/evaluation.py`
- Metrics: `rmse`, `mae`, `r2`, `sep` (SEP = std of residuals; bias-aware).
- `cross_validate_model(model, X, y, cv) -> dict[str, np.ndarray]` (per-fold metric arrays).
- `evaluate_on_test(model, X_tr, y_tr, X_te, y_te) -> dict[str, float]`.
- Plot helpers (save PNGs to `results/figures/`): `plot_spectra`, `plot_pred_vs_actual`,
  `plot_residuals`, `plot_pls_scan`, `plot_model_comparison`.

### `run_experiments.py` (repo root)
Driver that ties it together and produces real, reproducible artifacts:
1. Load data, EDA figures.
2. Compare preprocessing (raw vs SNV vs 2nd-deriv) for PLS.
3. PLS component scan.
4. Repeated-CV comparison of all models → metrics table (`results/cv_metrics.csv` + markdown).
5. Tune the best model on training set only.
6. Final evaluation on held-out test set → `results/test_metrics.json`.
7. Save all figures to `results/figures/`.

### `tecator_fat_regression.ipynb`
Narrative notebook mirroring the driver: intro → EDA → preprocessing comparison → model
comparison → tuning → final test eval → conclusions, with an explicit note on why the old
100% classification number was leakage-inflated. Executed end-to-end so outputs are real.

## 5. Testing

`tests/` with pytest:
- `test_data.py` — shapes, no NaNs, holdout sizes, stratification spans fat range, determinism.
- `test_preprocessing.py` — SNV rows ~zero-mean/unit-std; SavGol deriv shape preserved;
  transformers are sklearn-clonable and pipeline-safe (fit on train only).
- `test_evaluation.py` — metrics correct on toy data (rmse/mae/r2/sep known values).

## 6. Risks / Decisions

- **sklearn-only** (MLPRegressor) instead of a Keras 1D-CNN — keeps the repo lightweight and
  reproducible. Accepted.
- **Local CSV has 215 samples** (caret/FDA version) vs OpenML's 240 (with predefined splits).
  We use the 215-sample CSV already in the repo and create our own reproducible stratified
  holdout; we document this.
- Small N (215) → we rely on repeated CV (10×5) for stable estimates rather than one split.

## 7. Deliverables & Delivery

- `tecator/` package, `tests/`, `run_experiments.py`, `tecator_fat_regression.ipynb`,
  `results/` (metrics + figures), `requirements.txt`, visually appealing `README.md`.
- Original `challenge2.ipynb` retained for comparison.
- Committed and pushed to `main`.
