# Databricks notebook source
# MAGIC %md
# MAGIC # It's TIME — Slice 2: Zero-shot forecasting bake-off + pattern leaderboards
# MAGIC
# MAGIC Reproduces the **evaluation** half of *"It's TIME"* (ICML 2026) on Databricks. Consumes
# MAGIC `clean_series` + `variate_features` from Slice 1 and runs a **strict zero-shot** rolling-window
# MAGIC evaluation of:
# MAGIC
# MAGIC | model | kind | infra |
# MAGIC |---|---|---|
# MAGIC | **Seasonal-Naive** | probabilistic baseline (point + empirical residual quantiles) | none |
# MAGIC | **`ai_forecast()`** | Databricks built-in SQL forecaster | serverless SQL |
# MAGIC | **Chronos-Bolt** (`amazon/chronos-bolt-base`) | zero-shot TSFM | CPU (serverless) |
# MAGIC
# MAGIC Metrics follow the paper: **MASE** (point) and **CRPS/weighted-quantile-loss** (probabilistic),
# MAGIC each **normalised against Seasonal-Naive** and aggregated with the **geometric mean**. A
# MAGIC pattern-level breakdown (per F-bit, `Fk=1` vs `Fk=0`) mirrors the paper's Fig. 6.
# MAGIC
# MAGIC A clearly-flagged optional cell at the end shows how to scale up to **Chronos-2 / TimesFM-2.5** on
# MAGIC the T4 GPU cluster.

# COMMAND ----------

# MAGIC %pip install -q chronos-forecasting databricks-sql-connector
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "wscalioni_classic_catalog", "Catalog")
dbutils.widgets.text("schema", "its_time", "Schema")
dbutils.widgets.text("horizon", "24", "Forecast horizon H (steps)")
dbutils.widgets.text("n_windows", "3", "Rolling windows W")
dbutils.widgets.text("season_period", "24", "Seasonal period (steps)")
dbutils.widgets.text("frequency", "1 hour", "ai_forecast frequency string")
dbutils.widgets.text("warehouse_id", "228e047e9ad5595e", "SQL warehouse id (for ai_forecast)")
dbutils.widgets.text("gpu_models", "", "GPU models (comma list): Chronos-2,TimesFM-2.5")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
H = int(dbutils.widgets.get("horizon"))
W = int(dbutils.widgets.get("n_windows"))
SEASON = int(dbutils.widgets.get("season_period"))
FREQ = dbutils.widgets.get("frequency")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
GPU_MODELS = [m.strip() for m in dbutils.widgets.get("gpu_models").split(",") if m.strip()]
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{int(round(a * 100))}" for a in QLEVELS]  # q10..q90

print(f"{CATALOG}.{SCHEMA}  H={H}  W={W}  season={SEASON}  freq={FREQ}")

# COMMAND ----------

import numpy as np
import pandas as pd

clean = spark.table(f"{CATALOG}.{SCHEMA}.clean_series").toPandas().sort_values(["variate", "ts"])
feats = spark.table(f"{CATALOG}.{SCHEMA}.variate_features").toPandas()
variates = sorted(clean["variate"].unique())
print(f"{len(variates)} forecastable variates: {variates}")

# Build rolling test windows over the last W*H points; context = everything strictly before window
series = {v: clean.loc[clean.variate == v].reset_index(drop=True) for v in variates}
min_len = min(len(s) for s in series.values())
assert min_len > W * H + 2 * SEASON, "series too short for W*H test + context"

windows = []  # (window_idx, cutoff_pos) ; cutoff_pos = last context index (exclusive of window)
total = min_len
for w in range(W):
    start = total - (W - w) * H   # first test index of this window
    windows.append((w, start))
print("window test-start positions:", [s for _, s in windows])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics — MASE & weighted-quantile-loss (CRPS)

# COMMAND ----------

def mase(actual, pred, context, season):
    denom = np.mean(np.abs(context[season:] - context[:-season])) + 1e-9
    return float(np.mean(np.abs(actual - pred)) / denom)

def wql_crps(actual, qpreds, qlevels):
    """qpreds: dict level->array(H). Returns mean weighted-quantile-loss across levels (CRPS approx)."""
    denom = np.sum(np.abs(actual)) + 1e-9
    wqls = []
    for a in qlevels:
        q = qpreds[a]
        loss = np.where(actual < q, (a - 1) * (actual - q), a * (actual - q))
        wqls.append(2 * np.sum(loss) / denom)
    return float(np.mean(wqls))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1 — Seasonal-Naive (probabilistic)
# MAGIC Point = value one season back; quantiles = point + empirical quantiles of in-context
# MAGIC seasonal-naive residuals. This gives the baseline a distribution so CRPS can be normalised.

# COMMAND ----------

def seasonal_naive(context, H, season, qlevels):
    reps = int(np.ceil(H / season))
    base = np.tile(context[-season:], reps)[:H]
    resid = context[season:] - context[:-season]
    rq = {a: np.quantile(resid, a) for a in qlevels}
    qpreds = {a: base + rq[a] for a in qlevels}
    return base, qpreds  # point (median-ish), quantiles

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2 — Chronos-Bolt (zero-shot TSFM, CPU)

# COMMAND ----------

CHRONOS_OK = True
try:
    import torch
    from chronos import BaseChronosPipeline
    _pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    print("Chronos-Bolt loaded on CPU")
except Exception as e:
    CHRONOS_OK = False
    print("Chronos-Bolt unavailable, skipping:", repr(e))

def chronos_forecast(context, H, qlevels):
    # first positional arg is named `inputs` in current chronos-forecasting; pass positionally
    q, _mean = _pipe.predict_quantiles(
        torch.tensor(context, dtype=torch.float32),
        prediction_length=H, quantile_levels=qlevels)
    q = q[0].numpy()  # (H, len(qlevels))
    qpreds = {a: q[:, i] for i, a in enumerate(qlevels)}
    point = qpreds[0.5]
    return point, qpreds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 3 — `ai_forecast()` (Databricks built-in, point track)
# MAGIC One call per window across all series via `group_col`. The built-in returns a point forecast +
# MAGIC a single prediction interval, so it joins the **MASE** track (not the 9-quantile CRPS).
# MAGIC
# MAGIC ⚠️ `ai_forecast` is a preview AI function **disabled on serverless notebook (Spark Connect)
# MAGIC compute** in this workspace, but enabled on **SQL warehouses**. So we route this one call through
# MAGIC the serverless SQL warehouse via the SQL connector (auto-starts on first use).

# COMMAND ----------

AIF_OK = True
_wh_conn = None
try:
    from databricks import sql as dbsql
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _host = _ctx.apiUrl().get().replace("https://", "")
    _token = _ctx.apiToken().get()
    _wh_conn = dbsql.connect(server_hostname=_host,
                             http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
                             access_token=_token)
    print(f"SQL-warehouse connection open (warehouse {WAREHOUSE_ID})")
except Exception as e:
    AIF_OK = False
    print("ai_forecast warehouse connection unavailable, skipping:", repr(e)[:200])

def ai_forecast_window(cutoff_ts, horizon_end_ts):
    """Return dict variate -> point array(H). One ai_forecast call across all groups per window,
    executed on the SQL warehouse (where the preview function is enabled)."""
    sql = f"""
      SELECT variate, ts, value_forecast
      FROM ai_forecast(
        TABLE(SELECT variate, ts, value FROM {CATALOG}.{SCHEMA}.clean_series
              WHERE ts <= TIMESTAMP'{cutoff_ts}'),
        horizon => TIMESTAMP'{horizon_end_ts}',
        time_col => 'ts',
        value_col => 'value',
        group_col => 'variate',
        frequency => '{FREQ}',
        prediction_interval_width => 0.9)
      ORDER BY variate, ts
    """
    with _wh_conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    out = pd.DataFrame(rows, columns=["variate", "ts", "value_forecast"])
    res = {}
    for v, g in out.groupby("variate"):
        res[v] = g["value_forecast"].astype(float).to_numpy()[:H]
    return res

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 4 — GPU TSFMs (optional: Chronos-2, TimesFM-2.5)
# MAGIC Enabled only when the `gpu_models` widget is set **and** the notebook runs on the T4 GPU cluster.
# MAGIC Each loader is isolated in try/except so a package/API mismatch degrades to a skip. Every model
# MAGIC function has the uniform signature `fn(context, ts_slice) -> (point, qpreds)`.

# COMMAND ----------

gpu_fns = {}  # name -> fn(context, ts_slice) -> (point[H], {level: array[H]})

if "Chronos-2" in GPU_MODELS:
    try:
        import torch
        from chronos import Chronos2Pipeline
        _c2 = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

        def _c2_fn(context, ts_slice):
            cdf = pd.DataFrame({"id": "s", "timestamp": pd.to_datetime(ts_slice),
                                "target": np.asarray(context, dtype="float64")})
            pred = _c2.predict_df(cdf, prediction_length=H, quantile_levels=QLEVELS,
                                  id_column="id", timestamp_column="timestamp", target="target")
            qp = {}
            for a in QLEVELS:
                col = next((c for c in pred.columns
                            if _is_num(c) and abs(float(c) - a) < 1e-6), None)
                qp[a] = pred[col].to_numpy()[:H].astype(float)
            return qp[0.5], qp

        gpu_fns["Chronos-2"] = _c2_fn
        print("Chronos-2 loaded on GPU")
    except Exception as e:
        print("Chronos-2 unavailable, skipping:", repr(e)[:250])

if "TimesFM-2.5" in GPU_MODELS:
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timesfm[torch]"], check=True)
        import numpy as _np, timesfm
        _tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        _tfm.compile(timesfm.ForecastConfig(
            max_context=1024, max_horizon=max(256, H), normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))

        def _tfm_fn(context, ts_slice):
            pf, qf = _tfm.forecast(horizon=H, inputs=[_np.asarray(context, dtype="float32")])
            qf = _np.asarray(qf)[0]   # (H, 10): col0=mean, cols1..9 = Q0.1..Q0.9 (verified via probe)
            qp = {a: qf[:, i + 1].astype(float) for i, a in enumerate(QLEVELS)}
            return _np.asarray(pf)[0].astype(float), qp

        gpu_fns["TimesFM-2.5"] = _tfm_fn
        print("TimesFM-2.5 loaded on GPU")
    except Exception as e:
        print("TimesFM-2.5 unavailable, skipping:", repr(e)[:250])

print("GPU model functions available:", list(gpu_fns.keys()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the rolling backtest
# MAGIC Iterates a model registry; for every (model, variate, window) it records MASE + CRPS and the
# MAGIC per-step quantile predictions (→ `predictions` table, consumed by the forecast-inspector app).

# COMMAND ----------

def _is_num(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False

records = []      # per (model, variate, window): metrics
pred_rows = []    # per (model, variate, window, step): quantile predictions

def _add_pred_rows(model, v, w, ts_win, actual, point, qpreds):
    for i in range(len(actual)):
        row = dict(model=model, variate=v, window=int(w), step=i,
                   ts=pd.Timestamp(ts_win[i]), actual=float(actual[i]),
                   point=float(point[i]))
        for a, c in zip(QLEVELS, QCOLS):
            row[c] = float(qpreds[a][i]) if qpreds is not None else float("nan")
        pred_rows.append(row)

# pre-compute ai_forecast per window (batched across variates)
aif_cache = {}
if AIF_OK:
    any_v = variates[0]
    ts_index = series[any_v]["ts"].reset_index(drop=True)
    step = pd.Timestamp(ts_index.iloc[1]) - pd.Timestamp(ts_index.iloc[0])
    for w, start in windows:
        cutoff_ts = pd.Timestamp(ts_index.iloc[start - 1])
        # ai_forecast is endpoint-exclusive; request a few extra steps then slice to H
        horizon_end = cutoff_ts + step * (H + 2)
        try:
            aif_cache[w] = ai_forecast_window(str(cutoff_ts), str(horizon_end))
        except Exception as e:
            print(f"ai_forecast failed on window {w}: {repr(e)[:200]}")
            AIF_OK = False
            break

# Probabilistic model registry: name -> fn(context, ts_slice) -> (point, qpreds)
model_fns = {"Seasonal-Naive": lambda ctx, ts: seasonal_naive(ctx, H, SEASON, QLEVELS)}
if CHRONOS_OK:
    model_fns["Chronos-Bolt"] = lambda ctx, ts: chronos_forecast(ctx, H, QLEVELS)
model_fns.update(gpu_fns)  # Chronos-2, TimesFM-2.5 (if loaded)
print("Models in bake-off:", list(model_fns.keys()) + (["ai_forecast"] if AIF_OK else []))

for v in variates:
    s = series[v]["value"].to_numpy()
    ts_all = series[v]["ts"].to_numpy()
    for w, start in windows:
        context = s[:start]
        actual = s[start:start + H]
        ts_win = ts_all[start:start + H]
        if len(actual) < H:
            continue
        ctx_ts = ts_all[:start]
        for name, fn in model_fns.items():
            try:
                point, qpreds = fn(context, ctx_ts)
            except Exception as e:
                print(f"{name} failed on {v}/w{w}: {repr(e)[:150]}")
                continue
            records.append(dict(model=name, variate=v, window=w,
                                mase=mase(actual, point, context, SEASON),
                                crps=wql_crps(actual, qpreds, QLEVELS)))
            _add_pred_rows(name, v, w, ts_win, actual, point, qpreds)
        # ai_forecast (point track, quantiles NULL)
        if AIF_OK and w in aif_cache and v in aif_cache[w]:
            af_pt = aif_cache[w][v]
            if len(af_pt) == H:
                records.append(dict(model="ai_forecast", variate=v, window=w,
                                    mase=mase(actual, af_pt, context, SEASON), crps=np.nan))
                _add_pred_rows("ai_forecast", v, w, ts_win, actual, af_pt, None)

metrics = pd.DataFrame(records)
spark.createDataFrame(metrics).write.mode("overwrite").option(
    "mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.window_metrics")

# Per-step quantile predictions for the forecast-inspector app
pred_pdf = pd.DataFrame(pred_rows)
spark.createDataFrame(pred_pdf).write.mode("overwrite").option(
    "mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.predictions")

print(f"{len(metrics)} metric rows, {len(pred_pdf)} prediction rows")
display(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall leaderboard — normalised by Seasonal-Naive, geometric-mean aggregated
# MAGIC A task = (variate, window). Each model's metric is divided by the Seasonal-Naive metric on the
# MAGIC same task, then aggregated with the geometric mean (paper protocol). **< 1.0 beats the baseline.**

# COMMAND ----------

from scipy.stats import gmean

def normalized_leaderboard(metrics, metric_col):
    m = metrics.dropna(subset=[metric_col]).copy()
    base = (m[m.model == "Seasonal-Naive"][["variate", "window", metric_col]]
            .rename(columns={metric_col: "base"}))
    j = m.merge(base, on=["variate", "window"], how="inner")
    j["norm"] = j[metric_col] / (j["base"] + 1e-12)
    out = (j.groupby("model")["norm"].apply(lambda x: gmean(np.clip(x, 1e-6, None)))
           .rename(f"norm_{metric_col}_gmean").reset_index()
           .sort_values(f"norm_{metric_col}_gmean"))
    return out

lb_mase = normalized_leaderboard(metrics, "mase")
lb_crps = normalized_leaderboard(metrics, "crps")
leaderboard = lb_mase.merge(lb_crps, on="model", how="outer")
spark.createDataFrame(leaderboard).write.mode("overwrite").option(
    "mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.leaderboard")
print("Overall leaderboard (normalised vs Seasonal-Naive; <1 = better):")
display(leaderboard)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pattern-level breakdown (paper Fig. 6)
# MAGIC For each structural feature bit, compare each model's normalised MASE on variates **with** the
# MAGIC pattern (`Fk=1`) vs **without** (`Fk=0`). Reveals where a model's edge over the baseline lives.

# COMMAND ----------

bit_cols = ["F1_bit", "F2_bit", "F3_bit", "F4_bit", "F5_bit", "F6_bit", "F7_bit"]
fmap = feats.set_index("variate")[bit_cols]

m = metrics.dropna(subset=["mase"]).copy()
base = (m[m.model == "Seasonal-Naive"][["variate", "window", "mase"]]
        .rename(columns={"mase": "base"}))
j = m.merge(base, on=["variate", "window"]).merge(fmap, left_on="variate", right_index=True)
j["norm"] = j["mase"] / (j["base"] + 1e-12)

pat_rows = []
for bit in bit_cols:
    for grp in [1, 0]:
        sub = j[j[bit] == grp]
        for model, g in sub.groupby("model"):
            pat_rows.append(dict(feature=bit, group=grp, model=model,
                                 n=len(g), norm_mase_gmean=float(gmean(np.clip(g["norm"], 1e-6, None)))))
pattern_lb = pd.DataFrame(pat_rows)
spark.createDataFrame(pattern_lb).write.mode("overwrite").option(
    "mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.pattern_leaderboard")
display(pattern_lb.sort_values(["feature", "group", "model"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale up to Chronos-2 / TimesFM-2.5
# MAGIC The paper's top performers now run natively via the **Model 4** cell above — attach this notebook
# MAGIC to the T4 GPU cluster (`g4dn.xlarge`) and set the **`gpu_models`** widget to
# MAGIC `Chronos-2,TimesFM-2.5`. On serverless (widget empty) they're skipped and the bake-off runs
# MAGIC Seasonal-Naive + ai_forecast + Chronos-Bolt.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outputs
# MAGIC | table | contents |
# MAGIC |---|---|
# MAGIC | `window_metrics` | per (model, variate, window): MASE + CRPS |
# MAGIC | `predictions` | per (model, variate, window, step): `ts`, `actual`, `point`, `q10`…`q90` (app backing) |
# MAGIC | `leaderboard` | overall normalised-vs-Seasonal-Naive, geometric-mean (MASE + CRPS) |
# MAGIC | `pattern_leaderboard` | normalised MASE per feature-bit, `Fk=1` vs `Fk=0`, per model |
# MAGIC
# MAGIC These tables + `variate_features` back the **Slice 3** AI/BI leaderboard dashboard and the
# MAGIC Plotly-Dash forecast-inspector app (`predictions` + `clean_series`).
