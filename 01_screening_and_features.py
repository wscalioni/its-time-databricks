# Databricks notebook source
# MAGIC %md
# MAGIC # It's TIME — Slice 1: Data screening + structural tsfeatures
# MAGIC
# MAGIC Reproduces the **benchmark-construction** half of *"It's TIME: Towards the Next Generation of
# MAGIC Time Series Forecasting Benchmarks"* (ICML 2026) on Databricks:
# MAGIC
# MAGIC 1. **Automatic screening** — Algorithms 1 & 2 (timestamp/length/missing-rate, constant-series,
# MAGIC    Ljung-Box white-noise, sliding-window IQR outlier cleaning). Decides which series are even
# MAGIC    *forecastable*.
# MAGIC 2. **Structural `tsfeatures` F1–F7** via STL decomposition, then a **median-thresholded 7-bit
# MAGIC    pattern code** per variate — the retrieval key for pattern-level leaderboards in Slice 2.
# MAGIC
# MAGIC Ships with a **synthetic multi-pattern generator** (series with *known* shapes) so you can see the
# MAGIC pipeline recover trend / seasonality / noise / non-stationarity. A hook at the bottom shows how to
# MAGIC point it at a real Unity Catalog table instead.
# MAGIC
# MAGIC Runs on **serverless** notebook compute. Writes Delta tables to `{catalog}.{schema}`.

# COMMAND ----------

# MAGIC %pip install -q statsmodels scipy
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "wscalioni_classic_catalog", "Catalog")
dbutils.widgets.text("schema", "its_time", "Schema")
dbutils.widgets.dropdown("source", "synthetic", ["synthetic", "table"], "Data source")
dbutils.widgets.text("source_table", "", "Source table (if source=table): cat.sch.tbl")
dbutils.widgets.text("series_len", "1000", "Synthetic series length")
dbutils.widgets.text("season_period", "24", "Dominant seasonal period (steps)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SOURCE = dbutils.widgets.get("source")
SOURCE_TABLE = dbutils.widgets.get("source_table").strip()
L = int(dbutils.widgets.get("series_len"))
SEASON = int(dbutils.widgets.get("season_period"))

# Catalog is expected to already exist (managed-location/default-storage rules apply to CREATE CATALOG).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Target: {CATALOG}.{SCHEMA}  |  source={SOURCE}  L={L}  season={SEASON}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Data — synthetic multi-pattern generator
# MAGIC Each variate is engineered to a **known** structural signature so the pipeline's verdict can be
# MAGIC checked against ground truth. `predictable_truth` records what we expect the screener to conclude.

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
t = np.arange(L)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def make_synthetic(L, season):
    out = {}  # name -> (values, expected_predictable, note)
    # 1. Strong linear trend + mild noise
    out["strong_trend"] = (10 + 0.05 * t + rng.normal(0, 1, L), True, "trend-dominant")
    # 2. Strong pure seasonality (daily), stationary
    out["strong_seasonality"] = (
        20 + 6 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.6, L), True, "seasonal-dominant")
    # 3. Trend + seasonality (non-stationary, the classic real-world shape)
    out["trend_plus_season"] = (
        5 + 0.03 * t + 4 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.8, L),
        True, "trend+season, non-stationary")
    # 4. Stationary AR(1)
    ar = np.zeros(L)
    for i in range(1, L):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1)
    out["stationary_ar1"] = (50 + ar, True, "stationary AR(1)")
    # 5. White noise -> should be flagged NOT predictable (Ljung-Box)
    out["white_noise"] = (rng.normal(100, 5, L), False, "white noise -> unforecastable")
    # 6. Near-constant -> should be flagged NOT predictable (constant-series check)
    const = np.full(L, 7.0)
    const[rng.integers(0, L, 3)] += rng.normal(0, 0.01, 3)
    out["near_constant"] = (const, False, "near-constant -> unforecastable")
    # 7. Chaotic (logistic map) -> high complexity / spectral entropy
    chaos = np.zeros(L); chaos[0] = 0.4
    for i in range(1, L):
        chaos[i] = 3.99 * chaos[i - 1] * (1 - chaos[i - 1])
    out["chaotic_logistic"] = (100 * chaos, True, "chaotic, high complexity")
    # 8. Seasonal + injected extreme outliers -> IQR cleaning should catch spikes
    y = 30 + 5 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.7, L)
    idx = rng.integers(0, L, max(3, L // 200))
    y[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(40, 80, len(idx))
    out["seasonal_with_outliers"] = (y, True, "seasonal + extreme outliers")
    # 9. Trend + seasonality with missing values injected
    y = 8 + 0.02 * t + 3 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.7, L)
    miss = rng.random(L) < 0.05
    y[miss] = np.nan
    out["with_missing"] = (y, True, "5% missing -> imputed, still predictable")
    return out


if SOURCE == "synthetic":
    syn = make_synthetic(L, SEASON)
    base = pd.Timestamp("2024-01-01")
    rows = []
    truth = []
    for name, (vals, pred, note) in syn.items():
        ts = base + pd.to_timedelta(np.arange(len(vals)), unit="h")
        for tstamp, v in zip(ts, vals):
            rows.append(("synthetic", name, tstamp, float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else None))
        truth.append((name, pred, note))
    raw_pdf = pd.DataFrame(rows, columns=["dataset", "variate", "ts", "value"])
    truth_pdf = pd.DataFrame(truth, columns=["variate", "predictable_truth", "note"])
else:
    # source=table: expects columns dataset, variate, ts, value (long format)
    raw_pdf = spark.table(SOURCE_TABLE).select("dataset", "variate", "ts", "value").toPandas()
    truth_pdf = pd.DataFrame(columns=["variate", "predictable_truth", "note"])

raw_sdf = spark.createDataFrame(raw_pdf)
raw_sdf.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.raw_series")
print(f"raw_series: {raw_pdf.shape[0]} rows, {raw_pdf.variate.nunique()} variates")
display(truth_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Automatic screening — Algorithm 1 & 2
# MAGIC Per-variate quality check: numeric → length → missing-rate → constant-series → white-noise
# MAGIC (Ljung-Box) → sliding-window IQR outlier cleaning. Emits a **quality summary** and a cleaned series.

# COMMAND ----------

from statsmodels.stats.diagnostic import acorr_ljungbox

# Thresholds (paper defaults / sensible choices)
TAU_MISS = 0.30      # max missing rate
TAU_LEN = 3 * SEASON  # min length relative to frequency
TAU_EXT = 0.10       # max fraction of extreme outliers before series is dropped
K_EXT = 9            # IQR multiplier for extreme-outlier gate (paper sets k=9)


def normalized_entropy(x):
    vals, counts = np.unique(np.round(x, 6), return_counts=True)
    p = counts / counts.sum()
    if len(vals) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(vals)))


def topk_dominance(x, k=5):
    vals, counts = np.unique(np.round(x, 6), return_counts=True)
    order = np.argsort(counts)[::-1]
    return float(counts[order][:k].sum() / counts.sum())


def sliding_iqr_clean(x, window=None, k=K_EXT):
    """Return (cleaned, outlier_fraction). Local-window IQR filter; extremes -> forward fill."""
    L = len(x)
    window = window or max(2 * SEASON + 1, 25)
    half = window // 2
    xc = x.astype(float).copy()
    ext = np.zeros(L, dtype=bool)
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        w = x[lo:hi]
        q1, q3 = np.nanpercentile(w, 25), np.nanpercentile(w, 75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        m = np.nanmedian(w)
        if abs(x[i] - m) / iqr >= k:
            ext[i] = True
    frac = ext.mean()
    xc[ext] = np.nan
    xc = pd.Series(xc).ffill().bfill().to_numpy()
    return xc, float(frac)


def univariate_quality_check(x):
    reasons = []
    x = np.asarray(x, dtype="float64")
    L = len(x)
    # Check 2: length
    if L < TAU_LEN:
        return dict(predictable=False, reason="too_short", n_obs=L, missing_rate=None,
                    topk_dom=None, entropy=None, ljungbox_p=None, outlier_frac=None), x
    # missing rate
    miss = np.isnan(x)
    missing_rate = float(miss.mean())
    if missing_rate > TAU_MISS:
        return dict(predictable=False, reason="high_missing", n_obs=L, missing_rate=missing_rate,
                    topk_dom=None, entropy=None, ljungbox_p=None, outlier_frac=None), x
    # impute missing (ffill/bfill) before further tests
    xf = pd.Series(x).ffill().bfill().to_numpy()
    # Check 3: constant / degenerate signal
    dom = topk_dominance(xf)
    ent = normalized_entropy(xf)
    if dom >= 0.5 or ent < 0.1:
        return dict(predictable=False, reason="constant_series", n_obs=L, missing_rate=missing_rate,
                    topk_dom=dom, entropy=ent, ljungbox_p=None, outlier_frac=None), xf
    # Check 4: white-noise (Ljung-Box over lags 10 & 20; min p > 0.05 => white noise)
    try:
        lb = acorr_ljungbox(xf, lags=[10, 20], return_df=True)
        lb_p = float(lb["lb_pvalue"].min())
    except Exception:
        lb_p = 0.0
    if lb_p > 0.05:
        return dict(predictable=False, reason="white_noise", n_obs=L, missing_rate=missing_rate,
                    topk_dom=dom, entropy=ent, ljungbox_p=lb_p, outlier_frac=None), xf
    # Check 5: extreme-outlier detection + cleaning
    xc, ofrac = sliding_iqr_clean(xf)
    if ofrac > TAU_EXT:
        return dict(predictable=False, reason="excess_outliers", n_obs=L, missing_rate=missing_rate,
                    topk_dom=dom, entropy=ent, ljungbox_p=lb_p, outlier_frac=ofrac), xc
    return dict(predictable=True, reason="ok", n_obs=L, missing_rate=missing_rate,
                topk_dom=dom, entropy=ent, ljungbox_p=lb_p, outlier_frac=ofrac), xc


# Run screening per variate
summaries = []
clean_rows = []
for (ds, var), g in raw_pdf.sort_values("ts").groupby(["dataset", "variate"]):
    g = g.sort_values("ts")
    res, cleaned = univariate_quality_check(g["value"].to_numpy())
    res.update(dataset=ds, variate=var)
    summaries.append(res)
    if res["predictable"]:
        for tstamp, v in zip(g["ts"].to_numpy(), cleaned):
            clean_rows.append((ds, var, pd.Timestamp(tstamp), float(v)))

qual_pdf = pd.DataFrame(summaries)[
    ["dataset", "variate", "predictable", "reason", "n_obs", "missing_rate",
     "topk_dom", "entropy", "ljungbox_p", "outlier_frac"]]
spark.createDataFrame(qual_pdf).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.quality_summary")

clean_pdf = pd.DataFrame(clean_rows, columns=["dataset", "variate", "ts", "value"])
spark.createDataFrame(clean_pdf).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.clean_series")

# Ground-truth check (synthetic only)
if not truth_pdf.empty:
    chk = qual_pdf.merge(truth_pdf, on="variate", how="left")
    chk["match"] = chk["predictable"] == chk["predictable_truth"]
    print("Screening vs known ground truth:")
    display(chk[["variate", "note", "predictable_truth", "predictable", "reason", "match"]])
else:
    display(qual_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Structural tsfeatures F1–F7 (STL decomposition)
# MAGIC F1 Trend Strength · F2 Trend Linearity · F3 Seasonality Strength · F4 Seasonality Correlation ·
# MAGIC F5 Residual ACF-1 · F6 Complexity (spectral entropy) · F7 Stationarity (ADF).

# COMMAND ----------

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.signal import periodogram


def _var(a):
    return float(np.var(a, ddof=0))


def f1_trend_strength(T, R):
    return max(0.0, 1.0 - _var(R) / (_var(T + R) + 1e-12))


def f2_trend_linearity(T):
    n = len(T)
    tt = np.arange(n)
    P1 = tt - tt.mean()
    P1 = P1 / (np.linalg.norm(P1) + 1e-12)
    P2 = P1 ** 2 - (P1 ** 2).mean()
    P2 = P2 - P1 * (P1 @ P2)
    P2 = P2 / (np.linalg.norm(P2) + 1e-12)
    X = np.column_stack([np.ones(n), P1, P2])
    beta, *_ = np.linalg.lstsq(X, T, rcond=None)
    return float(beta[1])  # linear-term coefficient


def f3_seasonality_strength(S, R):
    return max(0.0, 1.0 - _var(R) / (_var(S + R) + 1e-12))


def f4_seasonality_correlation(S, period):
    K = len(S) // period
    if K < 2:
        return 0.0
    cyc = S[: K * period].reshape(K, period)
    cors = []
    for i in range(K):
        for j in range(i + 1, K):
            a, b = cyc[i], cyc[j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            cors.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cors)) if cors else 0.0


def f5_residual_acf1(R):
    R = R - R.mean()
    denom = np.sum(R ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(R[1:] * R[:-1]) / denom)


def f6_complexity(x):
    # normalized spectral entropy of the raw series
    _, pxx = periodogram(x - x.mean())
    pxx = pxx[1:]
    if pxx.sum() <= 0:
        return 0.0
    p = pxx / pxx.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def f7_stationarity(x):
    try:
        return int(adfuller(x, autolag="AIC")[1] < 0.05)
    except Exception:
        return 0


def compute_features(x, period):
    x = np.asarray(x, dtype="float64")
    # STL needs odd period >= 2 and >= 2 full cycles
    per = period if period % 2 == 1 else period + 1
    if len(x) < 2 * period or period < 2:
        T = pd.Series(x).rolling(max(3, period), min_periods=1, center=True).mean().to_numpy()
        S = np.zeros_like(x)
        R = x - T
    else:
        stl = STL(x, period=per, robust=True).fit()
        T, S, R = stl.trend, stl.seasonal, stl.resid
    return dict(
        F1_trend_strength=f1_trend_strength(T, R),
        F2_trend_linearity=f2_trend_linearity(T),
        F3_seasonality_strength=f3_seasonality_strength(S, R),
        F4_seasonality_correlation=f4_seasonality_correlation(S, period),
        F5_residual_acf1=f5_residual_acf1(R),
        F6_complexity=f6_complexity(x),
        F7_stationarity=f7_stationarity(x),
    )


feat_rows = []
for (ds, var), g in clean_pdf.sort_values("ts").groupby(["dataset", "variate"]):
    f = compute_features(g["value"].to_numpy(), SEASON)
    f.update(dataset=ds, variate=var)
    feat_rows.append(f)

feat_pdf = pd.DataFrame(feat_rows)
print(f"Computed F1-F7 for {len(feat_pdf)} predictable variates")
display(feat_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Median-thresholded 7-bit pattern code
# MAGIC F1–F6 are binarised at the **population median** (`Fk > median → 1`); F7 (stationarity) is already
# MAGIC binary. The concatenated 7-bit code is the retrieval key used to build pattern-level leaderboards.

# COMMAND ----------

cont = ["F1_trend_strength", "F2_trend_linearity", "F3_seasonality_strength",
        "F4_seasonality_correlation", "F5_residual_acf1", "F6_complexity"]
medians = feat_pdf[cont].median()
print("Population medians used as thresholds:\n", medians.to_string())

for c in cont:
    feat_pdf[c.split("_")[0] + "_bit"] = (feat_pdf[c] > medians[c]).astype(int)
feat_pdf["F7_bit"] = feat_pdf["F7_stationarity"].astype(int)

bit_cols = ["F1_bit", "F2_bit", "F3_bit", "F4_bit", "F5_bit", "F6_bit", "F7_bit"]
feat_pdf["pattern_code"] = feat_pdf[bit_cols].astype(str).agg("".join, axis=1)

spark.createDataFrame(feat_pdf).write.mode("overwrite").option(
    "mergeSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.variate_features")

display(feat_pdf[["variate"] + cont + ["F7_stationarity", "pattern_code"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Feature-distribution plot + recovered signatures
# MAGIC Sanity check: does the pipeline recover the engineered structure? e.g. `strong_trend` should have
# MAGIC high F1, `strong_seasonality` high F3/F4, `chaotic_logistic` high F6.

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, c in zip(axes.ravel(), cont):
    ax.hist(feat_pdf[c], bins=12, color="#FF3621", alpha=0.8)
    ax.axvline(medians[c], color="k", ls="--", lw=1, label="median")
    ax.set_title(c); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

print("\nRecovered signatures (variate -> pattern_code F1..F7):")
display(feat_pdf[["variate", "pattern_code",
                  "F1_trend_strength", "F3_seasonality_strength", "F6_complexity", "F7_stationarity"]]
        .sort_values("variate"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outputs
# MAGIC | table | contents |
# MAGIC |---|---|
# MAGIC | `raw_series` | ingested long-format series (dataset, variate, ts, value) |
# MAGIC | `quality_summary` | per-variate screening verdict + diagnostics |
# MAGIC | `clean_series` | forecastable variates only, outliers imputed |
# MAGIC | `variate_features` | F1–F7 + median-binarised bits + 7-bit `pattern_code` |
# MAGIC
# MAGIC ➡️ **Slice 2** (`02_zeroshot_bakeoff`) consumes `clean_series` + `variate_features` to run the
# MAGIC zero-shot model bake-off and build pattern-level leaderboards.
# MAGIC
# MAGIC ### Point it at real data
# MAGIC Set widget `source=table` and `source_table=cat.sch.tbl` where the table is long-format with
# MAGIC columns `dataset, variate, ts, value`. Adjust `season_period` to the dominant cycle of your data
# MAGIC (e.g. 24 for hourly-daily, 7 for daily-weekly, 12 for monthly-annual).
