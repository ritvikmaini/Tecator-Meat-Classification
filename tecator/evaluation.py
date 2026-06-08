"""Evaluation metrics, cross-validation runners, and plot helpers.

Used to score and visualise Tecator NIR fat-regression models.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so figures can be saved without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.base import clone
from sklearn.metrics import r2_score

__all__ = [
    "rmse",
    "mae",
    "r2",
    "sep",
    "cross_validate_model",
    "evaluate_on_test",
    "summarize_cv",
    "plot_spectra",
    "plot_pred_vs_actual",
    "plot_residuals",
    "plot_pls_scan",
    "plot_model_comparison",
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _as_1d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return y_true, y_pred


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true, y_pred):
    """Coefficient of determination (sklearn r2_score)."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(r2_score(y_true, y_pred))


def sep(y_true, y_pred):
    """Standard Error of Prediction: std of residuals (RMSE corrected for bias)."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    residuals = y_pred - y_true
    return float(np.std(residuals, ddof=1))


def _metric_dict(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "sep": sep(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Evaluation runners
# --------------------------------------------------------------------------- #
def cross_validate_model(model, X, y, cv):
    """Manually cross-validate ``model`` over ``cv.split(X)``.

    The model is cloned and refit on each training fold (no global fit) to
    avoid leakage. Returns a dict of per-fold metric arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    scores = {"rmse": [], "mae": [], "r2": [], "sep": []}
    for train_idx, test_idx in cv.split(X):
        est = clone(model)
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[test_idx])
        fold = _metric_dict(y[test_idx], y_pred)
        for key in scores:
            scores[key].append(fold[key])

    return {key: np.asarray(vals, dtype=np.float64) for key, vals in scores.items()}


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """Fit on train, predict test, return the 4 metrics plus ``y_pred``."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test), dtype=np.float64).ravel()

    result = _metric_dict(y_test, y_pred)
    result["y_pred"] = y_pred
    return result


def summarize_cv(results_by_model):
    """Summarise {model_name: cv_result_dict} into a sorted DataFrame."""
    rows = {}
    for name, res in results_by_model.items():
        rows[name] = {
            "rmse_mean": float(np.mean(res["rmse"])),
            "rmse_std": float(np.std(res["rmse"], ddof=1)) if len(res["rmse"]) > 1 else 0.0,
            "mae_mean": float(np.mean(res["mae"])),
            "r2_mean": float(np.mean(res["r2"])),
            "sep_mean": float(np.mean(res["sep"])),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[["rmse_mean", "rmse_std", "mae_mean", "r2_mean", "sep_mean"]]
    return df.sort_values("rmse_mean", ascending=True)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _finalize(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def plot_spectra(X, wavelengths, color_by=None, path=None):
    """Overlay all spectra vs wavelength, optionally colored by ``color_by``."""
    X = np.asarray(X, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64).ravel()

    fig, ax = plt.subplots(figsize=(9, 5))

    if color_by is not None:
        color_by = np.asarray(color_by, dtype=np.float64).ravel()
        norm = plt.Normalize(vmin=float(np.min(color_by)), vmax=float(np.max(color_by)))
        cmap = plt.get_cmap("viridis")
        segments = [np.column_stack([wavelengths, X[i]]) for i in range(X.shape[0])]
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.8, alpha=0.7)
        lc.set_array(color_by)
        ax.add_collection(lc)
        ax.autoscale()
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("color value")
    else:
        for i in range(X.shape[0]):
            ax.plot(wavelengths, X[i], linewidth=0.8, alpha=0.6)

    ax.set_title("NIR Spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.grid(True, alpha=0.3)
    return _finalize(fig, path)


def plot_pred_vs_actual(y_true, y_pred, path=None, title="Predicted vs Actual"):
    """Scatter of predicted vs actual with a y=x line and R2/RMSE annotation."""
    y_true, y_pred = _as_1d(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=25, alpha=0.7, edgecolor="k", linewidth=0.3)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")

    ax.annotate(
        f"$R^2$ = {r2(y_true, y_pred):.3f}\nRMSE = {rmse(y_true, y_pred):.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    return _finalize(fig, path)


def plot_residuals(y_true, y_pred, path=None):
    """Residuals (y_pred - y_true) vs predicted values."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=25, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)

    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred - actual)")
    ax.grid(True, alpha=0.3)
    return _finalize(fig, path)


def plot_pls_scan(n_components, rmse_values, path=None):
    """RMSE vs number of PLS components, marking the minimum."""
    n_components = np.asarray(n_components).ravel()
    rmse_values = np.asarray(rmse_values, dtype=np.float64).ravel()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_components, rmse_values, marker="o", linewidth=1.5)

    best = int(np.argmin(rmse_values))
    ax.scatter(
        [n_components[best]],
        [rmse_values[best]],
        color="red",
        zorder=5,
        s=80,
        label=f"min: {n_components[best]} comps (RMSE={rmse_values[best]:.3f})",
    )

    ax.set_title("PLS Component Scan")
    ax.set_xlabel("Number of PLS components")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _finalize(fig, path)


def plot_model_comparison(summary_df, path=None):
    """Horizontal bar chart of rmse_mean per model with rmse_std error bars."""
    df = summary_df.sort_values("rmse_mean", ascending=False)
    names = list(df.index)
    means = df["rmse_mean"].to_numpy()
    stds = df["rmse_std"].to_numpy() if "rmse_std" in df.columns else None

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(names) + 1)))
    ax.barh(names, means, xerr=stds, capsize=4, color="steelblue", alpha=0.85)

    ax.set_title("Model Comparison (CV RMSE)")
    ax.set_xlabel("RMSE (mean +/- std)")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.3)
    return _finalize(fig, path)
