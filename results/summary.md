# Tecator Fat Regression — Results

- Samples: **215** | 100 NIR channels (850–1050 nm) | fat range 0.9–49.1%
- Evaluation: repeated 5-fold CV (×10) on train; single held-out test (n=43), leakage-free pipelines.

## Cross-validation (sorted by RMSE)

|                             |   rmse_mean |   rmse_std |   mae_mean |   r2_mean |   sep_mean |
|:----------------------------|------------:|-----------:|-----------:|----------:|-----------:|
| GaussianProcess (2nd deriv) |      0.7368 |     0.3389 |     0.4846 |    0.9959 |     0.7357 |
| Stacking                    |      2.2093 |     0.5987 |     1.4575 |    0.9676 |     2.2018 |
| MLP                         |      2.3282 |     0.4835 |     1.7066 |    0.9643 |     2.2968 |
| PLS (SNV)                   |      2.4195 |     0.683  |     1.6575 |    0.96   |     2.4124 |
| PLS (raw)                   |      2.5796 |     0.5254 |     1.7824 |    0.9564 |     2.5606 |
| PCR                         |      2.7953 |     0.6258 |     1.9749 |    0.9491 |     2.7777 |
| PLS (2nd deriv)             |      2.9154 |     0.7438 |     1.9586 |    0.9428 |     2.9044 |
| KernelRidge (2nd deriv)     |      3.4357 |     1.6262 |     1.5208 |    0.9152 |     3.3256 |
| Ridge                       |      3.6421 |     0.3469 |     3.0516 |    0.9152 |     3.6285 |
| SVR (RBF)                   |      5.9745 |     1.0372 |     4.2084 |    0.7658 |     5.9593 |
| GradientBoosting            |      7.3041 |     1.1537 |     5.4253 |    0.6549 |     7.2945 |

## Selected model: **GaussianProcess (2nd deriv)**

- Held-out test: **RMSE 1.224**, MAE 0.841, **R² 0.9897**, SEP 1.087
- PLS 2nd-derivative scan best: 17 comps (CV RMSE 2.480) vs raw 15 comps (CV RMSE 2.349)
