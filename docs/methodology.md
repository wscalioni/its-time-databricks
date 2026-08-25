# Methodology — how the paper maps to this implementation

*It's TIME* (ICML 2026) is a **benchmark-and-methodology** paper for evaluating time-series foundation
models (TSFMs) under strict **zero-shot** conditions. It is not a new model; it is a rigorous harness plus
an argument about how forecasting evaluation should be done. This document maps each part of the paper to
the code in this repo.

## The four bottlenecks the paper attacks

1. **Legacy-constrained data coverage / contamination** — mainstream benchmarks recycle old public
   datasets (LSF, Monash, M4) that have likely leaked into TSFM pretraining corpora, inflating "zero-shot"
   scores. The paper curates 50 *fresh* datasets. *Here:* we generate a synthetic set with known
   signatures (no leakage possible, and ground truth is checkable), with a switch to real UC tables.
2. **Compromised data integrity** — little quality assurance in existing benchmarks. The paper adds a
   human-in-the-loop + automated screening pipeline. *Here:* Notebook 1, Algorithms 1 & 2.
3. **Misaligned task formulation** — one fixed horizon (e.g. 720 steps) for every dataset. The paper sets
   horizons per dataset from operational cadence + physical predictability. *Here:* per-run `horizon`
   with rolling windows in Notebook 2; the reasoning is surfaced in the guided app (Step 3).
4. **Limited analysis perspective** — a single scalar metric per domain hides *why* a model wins and can
   reward a smooth forecast that misses structure. The paper adds pattern-level stratification (Fig 6) and
   qualitative visual inspection (Figs 11–16). *Here:* `pattern_leaderboard` + the app's inspector.

## Stage 1 — Forecastability screening (Algorithms 1 & 2)

Implemented in `01_screening_and_features.py` (`univariate_quality_check`). A series is **kept** only if
it passes every check; otherwise it is dropped with a recorded reason (written to `quality_summary`).

| Check | Rule | Rationale |
|---|---|---|
| Type & length | numeric; `L ≥ 3 × season` | too-short series can't be evaluated |
| Missing rate | `ρ_miss = |{t: xₜ=NaN}| / L ≤ 0.30`; else fill | excessive gaps → unreliable |
| Constant / degenerate | drop if top-5 value dominance `≥ 0.5` **or** normalised entropy `< 0.1` | flat/near-constant series aren't forecasting problems |
| White noise (Ljung-Box) | drop if `min p_LB(lags 10,20) > 0.05` | noise is unforecastable by definition |
| Extreme outliers (sliding-window IQR) | flag points beyond `k·IQR` of the local median (`k=9`), replace; drop if too many | remove sensor spikes without deleting real dynamics |

**Ground-truth check:** the synthetic `white_noise` series fails Ljung-Box and `near_constant` fails the
constant check — exactly as intended; the other 7 pass.

## Stage 2 — Structural features F1–F7 (STL)

Each surviving variate is decomposed with **STL** (Seasonal-Trend via Loess): `x = T + S + R`. Seven
interpretable features (in `compute_features`), then a **median-thresholded 7-bit pattern code**:

| Feature | Definition |
|---|---|
| F1 Trend strength | `max(0, 1 − Var(R)/Var(T+R))` |
| F2 Trend linearity | linear coefficient of an orthogonal quadratic fit to `T` |
| F3 Seasonality strength | `max(0, 1 − Var(R)/Var(S+R))` |
| F4 Seasonality correlation | mean pairwise Pearson correlation between full seasonal cycles |
| F5 Residual ACF-1 | lag-1 autocorrelation of `R` |
| F6 Complexity | spectral entropy of the raw series (high = noise-like) |
| F7 Stationarity | ADF-test indicator (`p < 0.05 ⇒ 1`) |

F1–F6 are binarised at the **population median** (bit = 1 if above); F7 is already binary. The
concatenated 7 bits are the retrieval key for pattern-level analysis (`variate_features.pattern_code`).

Recovered signatures confirm the pipeline: `strong_trend` → high F1 / non-stationary; `strong_seasonality`
→ high F3; `chaotic_logistic` → high F6.

## Stage 3 — Forecasting task & horizons

`02_zeroshot_bakeoff.py` uses a **rolling-window** backtest: with horizon `H` and stride `H`, the tail of
each series is split into `W` non-overlapping windows; for each window the model sees only the history
*before* it and forecasts `H` steps. The paper chooses `H` per dataset from operational decision cadence
and physical predictability (human-driven series support longer horizons; chaotic physics-driven series
are capped). This repo exposes `horizon` and `n_windows` as widgets.

## Stage 4 — Zero-shot models

All models forecast without fine-tuning (see the models table in the top-level README). The
**Seasonal-Naive** baseline is made probabilistic by adding empirical quantiles of its in-sample seasonal
residuals, so it has a distribution for CRPS normalisation. `ai_forecast` participates in the point
(MASE) track only.

## Stage 5 — Metrics & normalisation

- **MASE** (point): `mean|Y−Ŷ| / mean|Yⱼ−Yⱼ₋ₛ|` (seasonal-naive-scaled).
- **CRPS** (probabilistic): mean **weighted quantile loss** over 9 levels,
  `wQL[α] = 2·Σ Λ_α(q̂,Y)/Σ|Y|`, `Λ_α(q,Y) = (α − 1{Y<q})(Y−q)`.
- **Normalisation**: each model's metric is divided by the Seasonal-Naive metric on the *same task*
  `(variate, window)`, then aggregated with the **geometric mean** (robust to outliers, symmetric in
  multiplicative gains). **< 1.0 beats the baseline.**

## Stage 6 — Pattern-level analysis (paper Fig 6)

`pattern_leaderboard` compares each model's normalised MASE on variates **with** a given feature (`Fk=1`)
vs **without** (`Fk=0`), exposing where a model's edge over the baseline actually lives.

## Stage 7 — Qualitative inspection (paper Figs 11–16)

Scalar metrics can hide a bad forecast (the paper shows a model scoring well while flattening a spike).
The Dash app's inspector shows the Q10–Q90 / Q30–Q70 predictive bands, median, and actuals in global and
local views so behaviour is inspected, not just scored.
