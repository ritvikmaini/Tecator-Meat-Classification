"""Tests for the NIR spectra preprocessing transformers."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tecator.preprocessing import (
    MultiplicativeScatterCorrection,
    SavitzkyGolay,
    StandardNormalVariate,
)

N_SAMPLES = 20
N_WAVELENGTHS = 100


@pytest.fixture
def spectra():
    """Synthetic smooth spectra with per-row scatter (offset + scale)."""
    rng = np.random.default_rng(42)
    wl = np.linspace(0, 4 * np.pi, N_WAVELENGTHS)
    base = np.sin(wl) + 0.5 * np.cos(2 * wl) + 3.0
    offsets = rng.normal(0, 1.0, size=(N_SAMPLES, 1))
    scales = rng.uniform(0.8, 1.2, size=(N_SAMPLES, 1))
    X = scales * base[None, :] + offsets
    X += rng.normal(0, 0.01, size=X.shape)
    return X


# --------------------------------------------------------------------------- #
# StandardNormalVariate
# --------------------------------------------------------------------------- #
def test_snv_rows_mean_zero_std_one(spectra):
    out = StandardNormalVariate().fit_transform(spectra)
    assert out.shape == (N_SAMPLES, N_WAVELENGTHS)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=1), 1.0, atol=1e-6)


def test_snv_fit_is_noop(spectra):
    t = StandardNormalVariate()
    assert t.fit(spectra) is t


# --------------------------------------------------------------------------- #
# SavitzkyGolay
# --------------------------------------------------------------------------- #
def test_savgol_shape_preserved(spectra):
    out = SavitzkyGolay().fit_transform(spectra)
    assert out.shape == (N_SAMPLES, N_WAVELENGTHS)


def test_savgol_deriv2_differs_from_deriv0(spectra):
    smooth = SavitzkyGolay(deriv=0).fit_transform(spectra)
    deriv2 = SavitzkyGolay(deriv=2).fit_transform(spectra)
    assert deriv2.shape == smooth.shape
    assert not np.allclose(smooth, deriv2)


def test_savgol_smoothing_reduces_noise():
    rng = np.random.default_rng(0)
    wl = np.linspace(0, 4 * np.pi, N_WAVELENGTHS)
    clean = np.sin(wl)[None, :].repeat(N_SAMPLES, axis=0)
    noisy = clean + rng.normal(0, 0.3, size=clean.shape)
    smoothed = SavitzkyGolay(window_length=11, polyorder=2, deriv=0).fit_transform(
        noisy
    )
    # High-frequency content measured via variance of successive differences.
    noisy_hf = np.var(np.diff(noisy, axis=1))
    smoothed_hf = np.var(np.diff(smoothed, axis=1))
    assert smoothed_hf < noisy_hf


# --------------------------------------------------------------------------- #
# MultiplicativeScatterCorrection
# --------------------------------------------------------------------------- #
def test_msc_shape_preserved(spectra):
    out = MultiplicativeScatterCorrection().fit_transform(spectra)
    assert out.shape == (N_SAMPLES, N_WAVELENGTHS)


def test_msc_reference_stored(spectra):
    t = MultiplicativeScatterCorrection().fit(spectra)
    assert hasattr(t, "reference_")
    np.testing.assert_allclose(t.reference_, spectra.mean(axis=0))


def test_msc_transforming_reference_returns_reference(spectra):
    t = MultiplicativeScatterCorrection().fit(spectra)
    ref = t.reference_.reshape(1, -1)
    out = t.transform(ref)
    np.testing.assert_allclose(out[0], t.reference_, atol=1e-6)


# --------------------------------------------------------------------------- #
# sklearn compatibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "transformer",
    [
        StandardNormalVariate(),
        MultiplicativeScatterCorrection(),
        SavitzkyGolay(window_length=7, polyorder=3, deriv=1),
    ],
)
def test_clonable(transformer):
    cloned = clone(transformer)
    assert cloned.get_params() == transformer.get_params()


@pytest.mark.parametrize(
    "transformer",
    [
        StandardNormalVariate(),
        MultiplicativeScatterCorrection(),
        SavitzkyGolay(),
    ],
)
def test_runs_in_pipeline(transformer, spectra):
    pipe = Pipeline([("pre", transformer), ("scaler", StandardScaler())])
    out = pipe.fit_transform(spectra)
    assert out.shape == (N_SAMPLES, N_WAVELENGTHS)
