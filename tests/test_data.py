import numpy as np
import pandas as pd

from tecator.data import WAVELENGTHS, load_data, make_holdout, repeated_cv


def test_load_data_shapes_and_targets():
    X, targets, wavelengths = load_data()

    assert X.shape == (215, 100)
    assert X.dtype == np.float64
    assert not np.isnan(X).any()

    assert isinstance(targets, pd.DataFrame)
    assert list(targets.columns) == ["fat", "water", "protein"]
    assert len(targets) == 215

    assert wavelengths is WAVELENGTHS or np.array_equal(wavelengths, WAVELENGTHS)


def test_wavelengths():
    assert WAVELENGTHS.shape == (100,)
    assert WAVELENGTHS.min() == 850.0
    assert WAVELENGTHS.max() == 1050.0


def test_make_holdout_sizes():
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()
    X_train, X_test, y_train, y_test = make_holdout(X, y)

    assert X_test.shape[0] == 43
    assert X_train.shape[0] == 172
    assert y_test.shape[0] == 43
    assert y_train.shape[0] == 172


def test_make_holdout_determinism():
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()

    a = make_holdout(X, y, random_state=42)
    b = make_holdout(X, y, random_state=42)

    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)


def test_make_holdout_test_spans_range():
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()
    _, _, _, y_test = make_holdout(X, y)

    assert y_test.min() >= y.min()
    assert y_test.max() <= y.max()
    # Not degenerate: the stratified test set should cover a wide fat span.
    assert y_test.std() > 5


def test_repeated_cv_fold_count():
    cv = repeated_cv(n_splits=5, n_repeats=10, random_state=42)
    X, targets, _ = load_data()
    y = targets["fat"].to_numpy()

    n_folds = sum(1 for _ in cv.split(X, y))
    assert n_folds == 50
