import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from tecator.evaluation import (
    cross_validate_model,
    mae,
    plot_pred_vs_actual,
    r2,
    rmse,
    sep,
)


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert np.isclose(r2(y, y), 1.0)


def test_known_small_example():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    residuals = y_pred - y_true
    expected_rmse = np.sqrt(np.mean(residuals**2))
    expected_mae = np.mean(np.abs(residuals))

    assert np.isclose(rmse(y_true, y_pred), expected_rmse)
    assert np.isclose(mae(y_true, y_pred), expected_mae)


def test_sep_matches_std_of_residuals():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    residuals = y_pred - y_true
    assert np.isclose(sep(y_true, y_pred), np.std(residuals, ddof=1))


def test_cross_validate_model_shapes_and_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.01 * rng.normal(size=40)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate_model(LinearRegression(), X, y, cv)

    for key in ("rmse", "mae", "r2", "sep"):
        assert key in results
        assert results[key].shape == (5,)
        assert np.all(np.isfinite(results[key]))


def test_plot_writes_nonempty_png(tmp_path):
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    out = tmp_path / "pred_vs_actual.png"

    fig = plot_pred_vs_actual(y_true, y_pred, path=str(out))

    assert fig is not None
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
