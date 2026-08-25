# Architecture

## Components

| Component | Technology | Role |
|---|---|---|
| Notebook 1 | Python on serverless (pandas, statsmodels, scipy) | screening + feature engineering |
| Notebook 2 | Python on serverless + T4 GPU (chronos-forecasting, timesfm, databricks-sql-connector) | zero-shot bake-off + metrics |
| Delta tables | Unity Catalog (`<catalog>.<schema>`) | all intermediate + result data |
| SQL warehouse | serverless (PRO) | `ai_forecast()` execution + dashboard/app queries |
| AI/BI dashboard | Lakeview | leaderboard + screening + feature summary |
| Guided-study app | Databricks Apps (Plotly Dash) | 8-step methodology walkthrough + inspector |

## Data flow

```
                 ┌───────────────────────── Notebook 1 (serverless) ─────────────────────────┐
 synthetic gen ─▶│ raw_series ─▶ screening (Alg 1&2) ─▶ clean_series ─▶ STL F1–F7 ─▶ codes    │
 (or UC table)   │                    │                                        │              │
                 │                    ▼                                        ▼              │
                 │              quality_summary                          variate_features     │
                 └────────────────────────────────────────────────────────────────────────── ┘
                                                       │ clean_series
                 ┌──────────────────────── Notebook 2 (serverless + GPU) ────────────────────┐
                 │ rolling windows ─▶ {Seasonal-Naive, ai_forecast*, Chronos-Bolt,            │
                 │                     Chronos-2†, TimesFM-2.5†} ─▶ MASE + CRPS                │
                 │   * via SQL warehouse   † via T4 GPU                                        │
                 │      │                        │                          │                  │
                 │      ▼                        ▼                          ▼                  │
                 │ window_metrics          leaderboard              pattern_leaderboard        │
                 │      │                                                                      │
                 │      ▼ (per-step quantiles)                                                 │
                 │  predictions                                                                │
                 └────────────────────────────────────────────────────────────────────────── ┘
                          │                                              │
                          ▼                                              ▼
                   AI/BI dashboard                             Guided-study Dash app
                   (leaderboard/*)                             (predictions + clean_series)
```

## Delta tables (`<catalog>.<schema>`)

| Table | Grain | Key columns |
|---|---|---|
| `raw_series` | series × timestamp | `dataset, variate, ts, value` |
| `quality_summary` | variate | `predictable, reason, n_obs, missing_rate, topk_dom, entropy, ljungbox_p, outlier_frac` |
| `clean_series` | forecastable variate × timestamp | `dataset, variate, ts, value` |
| `variate_features` | variate | `F1..F7` values, `F1_bit..F7_bit`, `pattern_code` |
| `window_metrics` | model × variate × window | `mase, crps` |
| `predictions` | model × variate × window × step | `ts, actual, point, q10..q90` |
| `leaderboard` | model | `norm_mase_gmean, norm_crps_gmean` |
| `pattern_leaderboard` | feature-bit × group × model | `n, norm_mase_gmean` |

## Compute & auth

- **Notebooks** authenticate through the notebook context; `ai_forecast()` is preview-disabled on
  serverless notebook (Spark Connect) compute, so Notebook 2 routes that one call through the serverless
  **SQL warehouse** via `databricks-sql-connector` (widget `warehouse_id`).
- **Dashboard** queries run on the SQL warehouse; the definition uses bare table names, with catalog +
  schema supplied at create time via `--dataset-catalog` / `--dataset-schema`.
- **App** connects to the SQL warehouse using its **service principal** identity (`Config()` +
  `credentials_provider`), with the warehouse attached as an app **resource** (`valueFrom: sql-warehouse`)
  and `SELECT` granted on the schema to the app SP.
