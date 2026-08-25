# It's TIME — reproduced on Databricks

A faithful, runnable reproduction of the benchmark-construction and evaluation methodology from
**"It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks"** (ICML 2026,
arXiv `2602.12147`) on the Databricks platform.

It rebuilds the paper's pipeline end-to-end — **forecastability screening → structural features →
zero-shot model bake-off → pattern-level analysis → qualitative inspection** — using Spark/Delta, the
built-in `ai_forecast()` function, open time-series foundation models (TSFMs), an AI/BI dashboard, and a
guided Databricks App. Everything runs on a **synthetic multi-pattern dataset with known signatures**, so
each stage's output is checkable against ground truth; a one-line switch points it at real Unity Catalog
tables instead.

Built and verified in the `fevm-wscalioni-classic` workspace.

---

## What's in here

| Path | What it is |
|---|---|
| `01_screening_and_features.py` | **Notebook 1** — data ingest, automatic screening (Algorithms 1 & 2), structural `tsfeatures` F1–F7 via STL, 7-bit pattern codes. |
| `02_zeroshot_bakeoff.py` | **Notebook 2** — strict zero-shot rolling-window evaluation of 5 models, MASE + CRPS, Seasonal-Naive normalisation, overall + pattern-level leaderboards, per-step quantile predictions. |
| `dashboard/its_time.json` | **AI/BI dashboard** — leaderboard, pattern-level bars, screening pie, feature table (serialized Lakeview definition). |
| `app/` | **Guided-study Databricks App** (Plotly Dash) — an 8-step walkthrough of the methodology ending in the quantile forecast inspector. |
| `docs/` | Methodology, architecture, deployment runbook, and results. |

### Documentation

- **[docs/methodology.md](docs/methodology.md)** — the paper's four bottlenecks and exactly how each part
  of the pipeline is implemented (screening checks, F1–F7 formulas, MASE/CRPS, normalisation).
- **[docs/architecture.md](docs/architecture.md)** — components, data flow, and the Delta tables produced.
- **[docs/deployment.md](docs/deployment.md)** — prerequisites and step-by-step deploy of the notebooks,
  dashboard, and app (CLI commands + the environment gotchas found during the build).
- **[docs/results.md](docs/results.md)** — verified results on the synthetic set (screening, features,
  5-model leaderboard).

---

## The pipeline at a glance

```
raw series ──▶ [NB1] screening (Alg 1&2) ──▶ clean_series ──▶ STL features F1–F7 ──▶ 7-bit pattern code
                     │                                                                      │
                     ▼                                                                      ▼
               quality_summary                                                       variate_features
                                                                                            │
clean_series ─▶ [NB2] rolling-window zero-shot bake-off (5 models) ─▶ MASE + CRPS ─▶ normalise vs
               Seasonal-Naive ─▶ geometric mean ─▶ leaderboard + pattern_leaderboard + predictions
                                                                    │
                                            ┌───────────────────────┴───────────────────────┐
                                            ▼                                                 ▼
                                   AI/BI dashboard                                  Guided-study Dash app
```

## Models evaluated (zero-shot)

| Model | Kind | Compute |
|---|---|---|
| Seasonal-Naive | probabilistic baseline (point + empirical residual quantiles) | none |
| `ai_forecast()` | Databricks built-in SQL forecaster | serverless SQL warehouse |
| Chronos-Bolt (`amazon/chronos-bolt-base`) | zero-shot TSFM | CPU / serverless |
| Chronos-2 (`amazon/chronos-2`) | zero-shot TSFM (multivariate-capable) | T4 GPU |
| TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`) | zero-shot TSFM | T4 GPU |

## Headline result (synthetic set, H=24, W=3)

Normalised MASE (geometric mean across tasks; **< 1.0 beats Seasonal-Naive**):

| Model | MASE | CRPS |
|---|---|---|
| Chronos-Bolt | 0.69 | 0.75 |
| TimesFM-2.5 | 0.70 | 0.76 |
| Chronos-2 | 0.71 | 0.76 |
| `ai_forecast` | 0.71 | — (point only) |
| Seasonal-Naive | 1.00 | 1.00 |

Every modern approach beats the naive baseline — the paper's central finding — and the platform built-in
`ai_forecast` lands right alongside the frontier TSFMs.

---

## Quickstart

1. **Prereqs**: a Databricks workspace with Unity Catalog, a serverless SQL warehouse, and (for the two
   GPU models) a small GPU cluster (T4/`g4dn.xlarge` is enough). See [docs/deployment.md](docs/deployment.md).
2. **Run Notebook 1** (`01_screening_and_features`) on serverless — writes `raw_series`,
   `quality_summary`, `clean_series`, `variate_features`.
3. **Run Notebook 2** (`02_zeroshot_bakeoff`) — on serverless it runs Seasonal-Naive + `ai_forecast` +
   Chronos-Bolt; set the `gpu_models` widget to `Chronos-2,TimesFM-2.5` and attach the GPU cluster to add
   the frontier TSFMs. Writes `window_metrics`, `predictions`, `leaderboard`, `pattern_leaderboard`.
4. **Deploy the dashboard and app** — see [docs/deployment.md](docs/deployment.md).

### Point it at real data

Notebook 1 widget `source=table`, `source_table=cat.sch.tbl` (long format: `dataset, variate, ts,
value`); set `season_period` to the dominant cycle (24 hourly-daily, 7 daily-weekly, 12 monthly-annual).

---

## Notes

- Not affiliated with the paper's authors; this is an independent reproduction for internal enablement.
- The synthetic generator and thresholds live in the notebooks and are fully documented inline.
- Provenance and per-decision rationale are in `docs/`.
