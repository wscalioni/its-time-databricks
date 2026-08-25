# Verified results

Reference run: synthetic multi-pattern set, `H=24`, `W=3`, 5 models. All numbers are reproducible by
re-running the two notebooks in any Unity Catalog workspace (set the `catalog`/`schema` widgets).

## Screening (Algorithms 1 & 2)

9 synthetic variates in; **7 kept, 2 rejected** — matching the engineered ground truth:

| variate | verdict | reason |
|---|---|---|
| strong_trend | kept | ok |
| strong_seasonality | kept | ok |
| trend_plus_season | kept | ok |
| stationary_ar1 | kept | ok |
| chaotic_logistic | kept | ok |
| seasonal_with_outliers | kept | ok (outliers cleaned) |
| with_missing | kept | ok (imputed) |
| **white_noise** | **rejected** | white_noise (Ljung-Box) |
| **near_constant** | **rejected** | constant_series |

## Structural features (recovered signatures)

| variate | F1 trend | F3 season | F6 complexity | F7 stationary | pattern_code |
|---|---|---|---|---|---|
| strong_trend | 0.99 | 0.25 | 0.28 | 0 | 1100000 |
| strong_seasonality | 0.03 | 0.98 | 0.23 | 1 | 0011101 |
| trend_plus_season | 0.99 | 0.94 | 0.34 | 0 | 1111000 |
| chaotic_logistic | 0.03 | 0.35 | 0.92 | 1 | 0000011 |
| stationary_ar1 | 0.28 | 0.28 | 0.81 | 1 | 0000111 |

The pipeline recovers the engineered structure: trend-dominant series score high F1 and are
non-stationary; seasonal series score high F3; the chaotic logistic map scores the highest F6.

## Leaderboard (normalised vs Seasonal-Naive, geometric mean; < 1.0 beats baseline)

| Model | MASE (norm) | CRPS (norm) |
|---|---|---|
| Chronos-Bolt | 0.693 | 0.752 |
| TimesFM-2.5 | 0.702 | 0.755 |
| Chronos-2 | 0.709 | 0.763 |
| ai_forecast | 0.709 | — (point track) |
| Seasonal-Naive | 1.000 | 1.000 |

`predictions` holds 504 rows per model (7 variates × 3 windows × 24 steps); all models carry Q10–Q90
quantiles except `ai_forecast` (point only).

## Takeaways (as framed for the AI Journal Club)

- Every modern approach beats the naive baseline by ~30% on normalised MASE — the paper's central claim,
  reproduced on independent data.
- The Databricks built-in `ai_forecast` is competitive with the frontier TSFMs on this set.
- Pattern-level analysis shows model edges shift by feature (e.g. behaviour differs on stationary vs
  non-stationary variates) — the reason a single leaderboard number is insufficient.
- The screening stage is a useful "don't forecast noise" gate in its own right, independent of any model.
