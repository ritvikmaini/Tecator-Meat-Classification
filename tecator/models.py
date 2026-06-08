"""Model definitions and tuning grids for the Tecator NIR fat-regression task.

Every model is a full scikit-learn :class:`~sklearn.pipeline.Pipeline` that
bundles any spectral preprocessing together with a ``StandardScaler`` and the
estimator. Keeping preprocessing inside the pipeline means it is fit only on the
training fold during cross-validation, so there is no information leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tecator.evaluation import cross_validate_model
from tecator.preprocessing import SavitzkyGolay, StandardNormalVariate

__all__ = ["build_models", "param_grids", "pls_component_scan", "vip_scores"]


class PLSRegressorRavel(PLSRegression):
    """PLSRegression that returns 1-D predictions.

    ``StackingRegressor`` expects base estimators to emit predictions of shape
    ``(n_samples,)``; vanilla :class:`~sklearn.cross_decomposition.PLSRegression`
    returns ``(n_samples, 1)``. This subclass simply ravels the output.
    """

    def predict(self, X, **kwargs):  # noqa: D401 - sklearn signature
        return super().predict(X, **kwargs).ravel()


def build_models(random_state=42):
    """Build the dictionary of named, leakage-safe regression pipelines.

    Parameters
    ----------
    random_state : int
        Seed used for the stochastic estimators (gradient boosting, MLP).

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
        Mapping from model name to a fully-specified pipeline. Pipeline step
        names are stable (``"snv"``, ``"savgol"``, ``"scaler"``, ``"pls"``,
        ``"pca"``, ``"ridge"``, ``"svr"`` ...) so they line up with the keys
        returned by :func:`param_grids`.
    """
    models = {}

    models["PLS (raw)"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=15)),
        ]
    )

    models["PLS (SNV)"] = Pipeline(
        [
            ("snv", StandardNormalVariate()),
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=15)),
        ]
    )

    models["PLS (2nd deriv)"] = Pipeline(
        [
            ("savgol", SavitzkyGolay(window_length=11, polyorder=2, deriv=2)),
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=15)),
        ]
    )

    models["PCR"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20)),
            ("linreg", LinearRegression()),
        ]
    )

    models["Ridge"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    models["SVR (RBF)"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, gamma="scale")),
        ]
    )

    models["GradientBoosting"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(random_state=random_state)),
        ]
    )

    models["MLP"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["KernelRidge (2nd deriv)"] = Pipeline(
        [
            ("savgol", SavitzkyGolay(window_length=11, polyorder=2, deriv=2)),
            ("scaler", StandardScaler()),
            ("kr", KernelRidge(kernel="rbf", alpha=0.01, gamma=None)),
        ]
    )

    models["GaussianProcess (2nd deriv)"] = Pipeline(
        [
            ("savgol", SavitzkyGolay(window_length=11, polyorder=2, deriv=2)),
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=ConstantKernel() * RBF() + WhiteKernel(),
                    normalize_y=True,
                    alpha=1e-6,
                    random_state=random_state,
                ),
            ),
        ]
    )

    stacking_estimators = [
        (
            "pls_d2",
            Pipeline(
                [
                    ("savgol", SavitzkyGolay(window_length=11, polyorder=2, deriv=2)),
                    ("scaler", StandardScaler()),
                    ("pls", PLSRegressorRavel(n_components=15)),
                ]
            ),
        ),
        (
            "svr",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel="rbf", C=100, gamma="scale")),
                ]
            ),
        ),
        (
            "kr_d2",
            Pipeline(
                [
                    ("savgol", SavitzkyGolay(window_length=11, polyorder=2, deriv=2)),
                    ("scaler", StandardScaler()),
                    ("kr", KernelRidge(kernel="rbf", alpha=0.01)),
                ]
            ),
        ),
    ]
    models["Stacking"] = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=False,
    )

    return models


def param_grids():
    """Return small, sane hyper-parameter grids keyed by model name.

    Keys match a subset of :func:`build_models` names. Param keys use the
    pipeline ``step__param`` convention with step names that exist in the
    corresponding pipeline, so the grids are ready to drop into
    ``GridSearchCV``.
    """
    return {
        "PLS (raw)": {"pls__n_components": [5, 10, 15, 20]},
        "PLS (SNV)": {"pls__n_components": [5, 10, 15, 20]},
        "PLS (2nd deriv)": {"pls__n_components": [5, 10, 15, 20]},
        "PCR": {"pca__n_components": [5, 10, 15, 20, 30]},
        "Ridge": {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
        "SVR (RBF)": {
            "svr__C": [1, 10, 100],
            "svr__gamma": ["scale", 0.01, 0.1],
        },
        "KernelRidge (2nd deriv)": {
            "kr__alpha": [0.001, 0.01, 0.1, 1.0],
            "kr__gamma": [None, 0.01, 0.1],
        },
    }


def vip_scores(pls_pipeline, X, y):
    """Variable Importance in Projection (VIP) scores for a PLS pipeline.

    The pipeline is fit on ``(X, y)``, the :class:`PLSRegression` step is located
    in ``named_steps``, and standard VIP scores are computed in the PLS input
    feature space from the fitted ``x_weights_``, ``x_scores_`` and
    ``y_loadings_``.

    Parameters
    ----------
    pls_pipeline : sklearn.pipeline.Pipeline
        A pipeline (fitted or unfitted) containing a ``PLSRegression`` step,
        optionally preceded by spectral preprocessing.
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)

    Returns
    -------
    numpy.ndarray
        1-D array of VIP scores, one per input feature to the PLS step
        (length ``p``).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    pls_pipeline.fit(X, y)

    pls = None
    for step in pls_pipeline.named_steps.values():
        if isinstance(step, PLSRegression):
            pls = step
            break
    if pls is None:
        raise ValueError("No PLSRegression step found in the pipeline.")

    t = np.asarray(pls.x_scores_)          # (n, A)
    w = np.asarray(pls.x_weights_)         # (p, A)
    q = np.asarray(pls.y_loadings_).ravel()  # (A,) robust to (1, A) or (A,)

    p, A = w.shape

    # SSY explained by each component: sum_samples(t_a^2) * q_a^2
    ssy = (t ** 2).sum(axis=0) * (q ** 2)  # (A,)
    total_ssy = ssy.sum()

    w_norm_sq = (w ** 2).sum(axis=0)       # (A,)
    w_norm_sq = np.where(w_norm_sq == 0.0, 1.0, w_norm_sq)

    # weight contribution per feature j: sum_a (w[j,a]^2 / ||w_a||^2) * ssy_a
    weighted = (w ** 2 / w_norm_sq) * ssy  # (p, A)
    vip = np.sqrt(p * weighted.sum(axis=1) / total_ssy)

    return vip


def pls_component_scan(X, y, cv, max_components=25, preprocessor=None):
    """Cross-validated RMSE as a function of the number of PLS components.

    For each ``n`` in ``1..max_components`` a pipeline of
    ``(optional preprocessor) -> StandardScaler -> PLSRegression(n_components=n)``
    is cross-validated with ``cv`` and the mean RMSE is recorded.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    cv : cross-validation splitter
        Anything with a ``split(X)`` method (e.g. ``KFold``, ``RepeatedKFold``).
    max_components : int
        Largest number of PLS latent variables to evaluate.
    preprocessor : sklearn transformer or None
        Optional spectral preprocessing step prepended to every pipeline.

    Returns
    -------
    components : list[int]
        ``[1, 2, ..., max_components]``.
    rmse_values : list[float]
        Mean cross-validated RMSE for each component count.
    """
    components = list(range(1, max_components + 1))
    rmse_values = []

    for n in components:
        steps = []
        if preprocessor is not None:
            steps.append(("preprocessor", preprocessor))
        steps.append(("scaler", StandardScaler()))
        steps.append(("pls", PLSRegression(n_components=n)))
        pipeline = Pipeline(steps)

        scores = cross_validate_model(pipeline, X, y, cv)
        rmse_values.append(float(np.mean(scores["rmse"])))

    return components, rmse_values
