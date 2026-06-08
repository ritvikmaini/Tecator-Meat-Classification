"""Tests for the best-in-class models and the VIP diagnostic."""

import numpy as np
import pytest

from tecator.data import load_data, make_holdout
from tecator.models import build_models, param_grids, vip_scores


@pytest.fixture(scope="module")
def train_subset():
    """Small slice of real Tecator training data for fast fitting."""
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()
    X_train, _, y_train, _ = make_holdout(X, y)
    # GP/stacking are a bit slower; ~80 rows keeps it fast but well-posed.
    return X_train[:80], y_train[:80]


NEW_MODELS = [
    "KernelRidge (2nd deriv)",
    "GaussianProcess (2nd deriv)",
    "Stacking",
]


def test_new_models_exist():
    models = build_models()
    for name in NEW_MODELS:
        assert name in models, f"{name} missing from build_models()"


@pytest.mark.parametrize("name", NEW_MODELS)
def test_new_models_fit_and_predict(train_subset, name):
    X, y = train_subset
    model = build_models()[name]
    model.fit(X, y)
    preds = np.asarray(model.predict(X)).ravel()
    assert preds.shape[0] == X.shape[0], f"{name}: wrong prediction length"
    assert np.all(np.isfinite(preds)), f"{name}: non-finite predictions"


def test_stacking_fits_without_error(train_subset):
    X, y = train_subset
    model = build_models()["Stacking"]
    model.fit(X, y)  # would raise if PLS 2D predictions broke stacking
    preds = np.asarray(model.predict(X)).ravel()
    assert np.all(np.isfinite(preds))


def test_kernelridge_param_grid():
    grids = param_grids()
    assert "KernelRidge (2nd deriv)" in grids
    grid = grids["KernelRidge (2nd deriv)"]
    assert grid["kr__alpha"] == [0.001, 0.01, 0.1, 1.0]
    assert grid["kr__gamma"] == [None, 0.01, 0.1]
    # No grids for GP or Stacking.
    assert "GaussianProcess (2nd deriv)" not in grids
    assert "Stacking" not in grids


def test_vip_scores_on_raw_pls(train_subset):
    X, y = train_subset
    pipeline = build_models()["PLS (raw)"]
    vip = vip_scores(pipeline, X, y)
    vip = np.asarray(vip)
    assert vip.ndim == 1
    assert vip.shape[0] == 100
    assert np.all(np.isfinite(vip))
    assert np.all(vip >= 0)
    # VIP is normalized so mean of squared VIP ~ 1.
    assert np.isclose(np.mean(vip ** 2), 1.0, atol=1e-6)
