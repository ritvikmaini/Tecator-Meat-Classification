"""End-to-end Tecator fat-regression experiments.

Produces reproducible artifacts under ``results/``:
  - figures/*.png        : EDA, PLS scan, model comparison, final fit plots
  - cv_metrics.csv       : repeated-CV metric summary per model
  - test_metrics.json    : held-out test metrics for the selected model
  - summary.md           : human-readable results summary

Run:  .venv/bin/python run_experiments.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold

from tecator.data import load_data, make_holdout, repeated_cv
from tecator.preprocessing import SavitzkyGolay
from tecator.models import build_models, param_grids, pls_component_scan, vip_scores
from tecator.evaluation import (
    cross_validate_model,
    evaluate_on_test,
    summarize_cv,
    plot_spectra,
    plot_pred_vs_actual,
    plot_residuals,
    plot_pls_scan,
    plot_model_comparison,
)

RESULTS = Path("results")
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("Tecator fat-content regression — experiments")
    print("=" * 70)

    # ---- Load + split -----------------------------------------------------
    X, targets, wavelengths = load_data()
    y = targets["fat"].to_numpy(dtype=np.float64)
    print(f"Loaded X={X.shape}, fat range [{y.min():.1f}, {y.max():.1f}], mean {y.mean():.2f}")

    X_train, X_test, y_train, y_test = make_holdout(X, y, test_size=43, random_state=RANDOM_STATE)
    print(f"Holdout: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # ---- EDA figures ------------------------------------------------------
    plot_spectra(X, wavelengths, color_by=y, path=FIGS / "spectra_by_fat.png")
    # 2nd-derivative spectra (the informative representation)
    deriv = SavitzkyGolay(window_length=11, polyorder=2, deriv=2).fit(X).transform(X)
    plot_spectra(deriv, wavelengths, color_by=y, path=FIGS / "spectra_2nd_derivative.png")
    print("Saved EDA figures.")

    # ---- PLS component scan (raw vs 2nd-derivative) -----------------------
    scan_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comps_raw, rmse_raw = pls_component_scan(
        X_train, y_train, scan_cv, max_components=25, preprocessor=None
    )
    comps_d2, rmse_d2 = pls_component_scan(
        X_train, y_train, scan_cv, max_components=25,
        preprocessor=SavitzkyGolay(window_length=11, polyorder=2, deriv=2),
    )
    plot_pls_scan(comps_d2, rmse_d2, path=FIGS / "pls_component_scan.png")

    # ---- VIP wavelength-importance diagnostic ----------------------------
    import matplotlib.pyplot as plt
    vip = vip_scores(build_models()["PLS (raw)"], X_train, y_train)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wavelengths, vip, color="darkgreen", linewidth=1.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="VIP = 1 (selection threshold)")
    ax.fill_between(wavelengths, 1.0, vip, where=(vip >= 1.0), alpha=0.2, color="green")
    ax.set_title("PLS Variable Importance in Projection (VIP) per wavelength")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("VIP score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "vip_scores.png", dpi=120)
    best_raw = int(comps_raw[int(np.argmin(rmse_raw))])
    best_d2 = int(comps_d2[int(np.argmin(rmse_d2))])
    print(f"PLS best n_components — raw: {best_raw} (RMSE {min(rmse_raw):.3f}) | "
          f"2nd-deriv: {best_d2} (RMSE {min(rmse_d2):.3f})")

    # ---- Repeated-CV model comparison ------------------------------------
    cv = repeated_cv(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    models = build_models(random_state=RANDOM_STATE)
    cv_results = {}
    for name, model in models.items():
        res = cross_validate_model(model, X_train, y_train, cv)
        cv_results[name] = res
        print(f"  {name:18s} RMSE {res['rmse'].mean():.3f} ± {res['rmse'].std():.3f} | "
              f"R2 {res['r2'].mean():.4f}")

    summary = summarize_cv(cv_results)
    summary.to_csv(RESULTS / "cv_metrics.csv")
    plot_model_comparison(summary, path=FIGS / "model_comparison.png")
    print("\nCV summary (sorted by RMSE):")
    print(summary.round(4).to_string())

    # ---- Select best model, tune on training set only --------------------
    best_name = summary.index[0]
    best_model = clone(models[best_name])
    grids = param_grids()
    tuned_info = {"model": best_name, "tuned": False}
    if best_name in grids:
        gs = GridSearchCV(
            clone(models[best_name]), grids[best_name],
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        tuned_info = {"model": best_name, "tuned": True,
                      "best_params": gs.best_params_,
                      "cv_best_rmse": float(-gs.best_score_)}
        print(f"\nTuned {best_name}: {gs.best_params_} (CV RMSE {-gs.best_score_:.3f})")

    # ---- Final held-out test evaluation ----------------------------------
    test_res = evaluate_on_test(best_model, X_train, y_train, X_test, y_test)
    y_pred = test_res.pop("y_pred")
    plot_pred_vs_actual(y_test, y_pred, path=FIGS / "final_pred_vs_actual.png",
                        title=f"Held-out test — {best_name}")
    plot_residuals(y_test, y_pred, path=FIGS / "final_residuals.png")
    print(f"\nHELD-OUT TEST ({best_name}): "
          f"RMSE {test_res['rmse']:.3f} | MAE {test_res['mae']:.3f} | "
          f"R2 {test_res['r2']:.4f} | SEP {test_res['sep']:.3f}")

    # ---- Persist machine-readable + markdown summary ---------------------
    out = {
        "selected_model": best_name,
        "tuning": tuned_info,
        "test_metrics": test_res,
        "cv_best": {
            "model": best_name,
            "rmse_mean": float(summary.loc[best_name, "rmse_mean"]),
            "rmse_std": float(summary.loc[best_name, "rmse_std"]),
            "r2_mean": float(summary.loc[best_name, "r2_mean"]),
        },
        "pls_scan": {"raw_best_comp": best_raw, "raw_best_rmse": float(min(rmse_raw)),
                     "d2_best_comp": best_d2, "d2_best_rmse": float(min(rmse_d2))},
        "fat_range": [float(y.min()), float(y.max())],
        "n_samples": int(X.shape[0]),
    }
    with open(RESULTS / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    _write_summary_md(summary, out)
    print("\nWrote results/cv_metrics.csv, results/test_metrics.json, results/summary.md")
    print("Done.")
    return out


def _write_summary_md(summary, out):
    lines = ["# Tecator Fat Regression — Results\n"]
    lines.append(f"- Samples: **{out['n_samples']}** | 100 NIR channels (850–1050 nm) | "
                 f"fat range {out['fat_range'][0]:.1f}–{out['fat_range'][1]:.1f}%")
    lines.append(f"- Evaluation: repeated 5-fold CV (×10) on train; single held-out test "
                 f"(n=43), leakage-free pipelines.\n")
    lines.append("## Cross-validation (sorted by RMSE)\n")
    disp = summary.copy().round(4)
    lines.append(disp.to_markdown())
    sel = out["selected_model"]
    t = out["test_metrics"]
    lines.append(f"\n## Selected model: **{sel}**\n")
    if out["tuning"].get("tuned"):
        lines.append(f"- Tuned params: `{out['tuning']['best_params']}`")
    lines.append(f"- Held-out test: **RMSE {t['rmse']:.3f}**, MAE {t['mae']:.3f}, "
                 f"**R² {t['r2']:.4f}**, SEP {t['sep']:.3f}")
    lines.append(f"- PLS 2nd-derivative scan best: {out['pls_scan']['d2_best_comp']} comps "
                 f"(CV RMSE {out['pls_scan']['d2_best_rmse']:.3f}) vs raw "
                 f"{out['pls_scan']['raw_best_comp']} comps "
                 f"(CV RMSE {out['pls_scan']['raw_best_rmse']:.3f})")
    Path("results/summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
