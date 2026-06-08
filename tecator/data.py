"""Data loading and splitting utilities for the Tecator NIR fat-regression task."""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, train_test_split

_TARGET_COLS = ["fat", "water", "protein"]


def _read_raw(path):
    df = pd.read_csv(path)
    spectrum_cols = [c for c in df.columns if c not in _TARGET_COLS]
    return df, spectrum_cols


def _parse_wavelengths(spectrum_cols):
    values = []
    for col in spectrum_cols:
        match = re.search(r"[-+]?\d*\.?\d+", col)
        if match is None:
            raise ValueError(f"Cannot parse wavelength from column name: {col!r}")
        values.append(float(match.group()))
    return np.asarray(values, dtype=np.float64)


# Parse the canonical wavelength grid from the dataset header so it always
# matches the spectrum columns used by load_data.
_DEFAULT_PATH = "data/tecator.csv"
_raw, _spectrum_cols = _read_raw(_DEFAULT_PATH)
WAVELENGTHS = _parse_wavelengths(_spectrum_cols)


def load_data(path=_DEFAULT_PATH):
    """Load Tecator spectra and targets.

    Returns
    -------
    X : np.ndarray, shape (n, 100), float64
        The NIR absorbance spectra (wavelength columns only).
    targets : pd.DataFrame
        DataFrame with columns ``fat``, ``water``, ``protein``.
    wavelengths : np.ndarray
        The WAVELENGTHS array (length 100).
    """
    df, spectrum_cols = _read_raw(path)
    X = df[spectrum_cols].to_numpy(dtype=np.float64)
    targets = df[_TARGET_COLS].copy()
    wavelengths = _parse_wavelengths(spectrum_cols)
    return X, targets, wavelengths


def make_holdout(X, y, test_size=43, random_state=42):
    """Create a reproducible holdout split stratified by fat quantile bins.

    Parameters
    ----------
    X : array-like, shape (n, n_features)
    y : array-like, shape (n,)
        The fat values used both as the regression target and for binning.
    test_size : int
        Number of samples in the test set (default 43).
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Stratify on quantile bins so the test set spans the full fat range.
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=bins,
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


def repeated_cv(n_splits=5, n_repeats=10, random_state=42):
    """Return a RepeatedKFold cross-validator (n_splits * n_repeats folds)."""
    return RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
