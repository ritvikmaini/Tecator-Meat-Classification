"""Tests for tecator.models: pipelines, param grids, and the PLS scan."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from tecator.data import load_data, make_holdout
from tecator.models import build_models, param_grids, pls_component_scan


@pytest.fixture(scope="module")
def train_subset():
    """A small slice of real Tecator training data for fast fitting."""
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()
    X_train, _, y_train, _ = make_holdout(X, y)
    # Subset for speed; keep enough rows so PLS(20)/PCA(20) are well-posed.
    return X_train[:60], y_train[:60]


def test_build_models_returns_dict():
    models = build_models()
    assert isinstance(models, dict)
    assert len(models) >= 8
    for est in models.values():
        # Most models are Pipelines; the Stacking ensemble is a plain estimator
        # whose bases handle their own preprocessing.
        assert isinstance(est, (Pipeline, BaseEstimator))


def test_models_fit_and_predict(train_subset):
    X, y = train_subset
    models = build_models()
    for name, pipe in models.items():
        pipe.fit(X, y)
        preds = pipe.predict(X)
        preds = np.asarray(preds).ravel()
        assert preds.shape[0] == X.shape[0], f"{name}: wrong prediction length"
        assert np.all(np.isfinite(preds)), f"{name}: non-finite predictions"


def test_param_grids_keys_subset_of_models():
    models = build_models()
    grids = param_grids()
    assert set(grids).issubset(set(models))


def test_param_grid_prefixes_match_pipeline_steps():
    models = build_models()
    grids = param_grids()
    for name, grid in grids.items():
        steps = set(models[name].named_steps)
        for param_key in grid:
            prefix = param_key.split("__")[0]
            assert prefix in steps, (
                f"{name}: param prefix {prefix!r} not in steps {steps}"
            )


def test_pls_component_scan(train_subset):
    X, y = train_subset
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    max_components = 8
    components, rmse_values = pls_component_scan(
        X, y, cv, max_components=max_components
    )
    assert len(components) == max_components
    assert len(rmse_values) == max_components
    assert len(components) == len(rmse_values)
    assert components == list(range(1, max_components + 1))
    assert all(np.isfinite(v) for v in rmse_values)


def test_pls_component_scan_with_preprocessor(train_subset):
    from tecator.preprocessing import StandardNormalVariate

    X, y = train_subset
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    components, rmse_values = pls_component_scan(
        X, y, cv, max_components=5, preprocessor=StandardNormalVariate()
    )
    assert len(components) == 5
    assert all(np.isfinite(v) for v in rmse_values)
