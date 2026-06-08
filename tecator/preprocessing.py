"""Scatter-correction and smoothing transformers for NIR spectra.

Each transformer is scikit-learn compatible (subclasses ``BaseEstimator`` and
``TransformerMixin``), operates on a spectra matrix ``X`` of shape
``(n_samples, n_wavelengths)`` and returns an array of the same shape. They are
safe to use inside an sklearn ``Pipeline``: ``fit`` only learns from the data
passed to it (no global/leaky state).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = [
    "StandardNormalVariate",
    "MultiplicativeScatterCorrection",
    "SavitzkyGolay",
]


class StandardNormalVariate(BaseEstimator, TransformerMixin):
    """Row-wise Standard Normal Variate (SNV) correction.

    For each spectrum (row) subtract its mean and divide by its standard
    deviation. This is a stateless, per-row operation, so ``fit`` is a no-op.
    """

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        check_array(X)
        return self

    def transform(self, X):
        X = check_array(X, dtype=np.float64)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        # Avoid division by zero for flat spectra.
        std = np.where(std == 0.0, 1.0, std)
        return (X - mean) / std


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction (MSC).

    ``fit`` stores a reference spectrum (the mean spectrum across the rows of
    the training data) in ``self.reference_``. ``transform`` regresses each row
    against the reference (``row = a + b * reference`` via a degree-1
    least-squares fit) and returns ``(row - a) / b``.
    """

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        X = check_array(X, dtype=np.float64)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "reference_")
        X = check_array(X, dtype=np.float64)
        reference = self.reference_
        corrected = np.empty_like(X)
        for i, row in enumerate(X):
            # row = a + b * reference  ->  polyfit(reference, row, 1) = [b, a]
            b, a = np.polyfit(reference, row, 1)
            if np.abs(b) < 1e-12:
                # Slope ~0: scatter correction is undefined; fall back to a
                # mean-centred row to avoid blowing up the values.
                corrected[i] = row - a
            else:
                corrected[i] = (row - a) / b
        return corrected


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing / differentiation along the wavelength axis.

    Applies :func:`scipy.signal.savgol_filter` along ``axis=1``. ``deriv=1`` or
    ``deriv=2`` produce derivative spectra. ``fit`` is a no-op.

    Parameters are stored unmodified on the instance under the exact names
    ``window_length``, ``polyorder`` and ``deriv`` so that sklearn's
    ``get_params``/``clone`` machinery works correctly.
    """

    def __init__(self, window_length=11, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        check_array(X)
        return self

    def transform(self, X):
        X = check_array(X, dtype=np.float64)
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )
