"""It's TIME — Guided Study (Databricks App, Plotly Dash).

A hand-holding, technical walkthrough of the *It's TIME* (ICML 2026) forecasting-benchmark
methodology. Eight Prev/Next steps, each pairing a paper explanation with a live visual on the tables
produced by the notebooks (catalog/schema from ITS_TIME_CATALOG/ITS_TIME_SCHEMA env vars, default
`main`). The final step is the quantile forecast inspector.
"""
import os
import functools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from databricks.sdk.core import Config
from databricks import sql

CATALOG = os.getenv("ITS_TIME_CATALOG", "main")
SCHEMA = os.getenv("ITS_TIME_SCHEMA", "its_time")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
SEASON = int(os.getenv("ITS_TIME_SEASON", "24"))
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
BAND = "rgba(34,114,180,{a})"
ACCENT = "#2272B4"
_cfg = Config()

# ---------------------------------------------------------------- data access
def _conn():
    return sql.connect(
        server_hostname=_cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: _cfg.authenticate,
    )


def q(sql_text):
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql_text)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


@functools.lru_cache(maxsize=1)
def dropdown_options():
    d = q(f"SELECT DISTINCT variate, model, window FROM {CATALOG}.{SCHEMA}.predictions")
    variates = sorted(d["variate"].unique())
    models = sorted(d["model"].unique())
    windows = sorted(int(w) for w in d["window"].unique())
    return variates, models, windows


def preds(variate, model, window):
    return q(f"""SELECT step, ts, actual, point, {', '.join(QCOLS)}
                 FROM {CATALOG}.{SCHEMA}.predictions
                 WHERE variate='{variate}' AND model='{model}' AND window={window}
                 ORDER BY step""")


def full_series(variate):
    return q(f"""SELECT ts, value FROM {CATALOG}.{SCHEMA}.clean_series
                 WHERE variate='{variate}' ORDER BY ts""")


def raw_series(variate):
    return q(f"""SELECT ts, value FROM {CATALOG}.{SCHEMA}.raw_series
                 WHERE variate='{variate}' ORDER BY ts""")


@functools.lru_cache(maxsize=1)
def quality_df():
    return q(f"""SELECT variate, predictable, reason, n_obs, missing_rate, topk_dom, entropy,
                        ljungbox_p, outlier_frac
                 FROM {CATALOG}.{SCHEMA}.quality_summary ORDER BY predictable DESC, variate""")


@functools.lru_cache(maxsize=1)
def leaderboard_df():
    return q(f"SELECT model, norm_mase_gmean, norm_crps_gmean FROM {CATALOG}.{SCHEMA}.leaderboard ORDER BY norm_mase_gmean")


def feature_row(variate):
    return q(f"SELECT * FROM {CATALOG}.{SCHEMA}.variate_features WHERE variate='{variate}'")


def metrics_for(variate):
    return q(f"SELECT model, window, mase, crps FROM {CATALOG}.{SCHEMA}.window_metrics WHERE variate='{variate}'")


def windows_for(variate):
    return q(f"""SELECT window, min(ts) AS t0, max(ts) AS t1
                 FROM {CATALOG}.{SCHEMA}.predictions WHERE variate='{variate}'
                 GROUP BY window ORDER BY window""")


def overlay_for(variate, window):
    return q(f"""SELECT model, ts, point, actual FROM {CATALOG}.{SCHEMA}.predictions
                 WHERE variate='{variate}' AND window={window} ORDER BY model, ts""")


def stl_decompose(variate):
    from statsmodels.tsa.seasonal import STL
    s = full_series(variate)
    s["ts"] = pd.to_datetime(s["ts"])
    vals = s["value"].to_numpy(dtype="float64")
    per = SEASON if SEASON % 2 == 1 else SEASON + 1
    r = STL(vals, period=per, robust=True).fit()
    return s["ts"], vals, r.trend, r.seasonal, r.resid


# ---------------------------------------------------------------- shared UI bits
def _empty(msg):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=15))
    f.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=20, b=30))
    return f


def _graph(fig, height="52vh"):
    return dcc.Graph(figure=fig, style={"height": height})


def _table(df, fmt=None):
    d = df.copy()
    for c in d.columns:
        if fmt and c in fmt:
            d[c] = d[c].map(fmt[c])
    return dbc.Table.from_dataframe(d, striped=True, bordered=False, hover=True,
                                    size="sm", class_name="mb-0")


def _base_layout(fig, title):
    fig.update_layout(template="plotly_white",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
                      margin=dict(l=40, r=20, t=48, b=60),
                      title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top"))
    return fig


# ---------------------------------------------------------------- step visuals
def viz_leaderboard(highlight_best=True):
    lb = leaderboard_df().copy()
    lb["norm_mase_gmean"] = lb["norm_mase_gmean"].astype(float)
    colors = [("#FFA600" if highlight_best and i == 0 else ACCENT) for i in range(len(lb))]
    fig = go.Figure(go.Bar(x=lb["model"], y=lb["norm_mase_gmean"], marker_color=colors,
                           text=lb["norm_mase_gmean"].round(3), textposition="outside"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                  annotation_text="Seasonal-Naive baseline (1.0)")
    return _base_layout(fig, "Normalised MASE by model (lower = better)")


def viz_screening(variate):
    r = raw_series(variate)
    r["ts"] = pd.to_datetime(r["ts"])
    qd = quality_df()
    row = qd[qd["variate"] == variate].iloc[0]
    verdict = "PREDICTABLE" if str(row["predictable"]).lower() in ("true", "1") else f"REJECTED ({row['reason']})"
    color = ACCENT if "PREDICT" in verdict else "#FF7054"
    fig = go.Figure(go.Scatter(x=r["ts"], y=r["value"], line=dict(color=color, width=1)))
    return _base_layout(fig, f"{variate} — screening verdict: {verdict}")


def viz_stl(variate):
    ts, vals, T, S, R = stl_decompose(variate)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed x", "Trend T", "Seasonal S", "Remainder R"),
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=ts, y=vals, line=dict(color="#11171C", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=T, line=dict(color=ACCENT, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=S, line=dict(color="#DE5582", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=R, line=dict(color="#999", width=1)), row=4, col=1)
    fig.update_layout(template="plotly_white", showlegend=False,
                      margin=dict(l=40, r=20, t=30, b=30),
                      title=dict(text=f"STL decomposition — {variate}", x=0.01, y=0.99, yanchor="top"))
    return fig


def viz_windows(variate):
    s = full_series(variate)
    s["ts"] = pd.to_datetime(s["ts"])
    fig = go.Figure(go.Scatter(x=s["ts"], y=s["value"], name="series", line=dict(color="#9AA7B2", width=1)))
    w = windows_for(variate)
    for _, wr in w.iterrows():
        fig.add_vrect(x0=pd.to_datetime(wr["t0"]), x1=pd.to_datetime(wr["t1"]),
                      fillcolor="#FFA600", opacity=0.15, line_width=0,
                      annotation_text=f"window {int(wr['window'])}", annotation_position="top left")
    return _base_layout(fig, f"{variate} — {len(w)} rolling test windows (each H steps)")


def viz_overlay(variate, window):
    d = overlay_for(variate, window)
    d["ts"] = pd.to_datetime(d["ts"])
    fig = go.Figure()
    actual = d[d["model"] == "Seasonal-Naive"][["ts", "actual"]]
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["ts"], y=actual["actual"].astype(float), name="Actual",
                                 line=dict(color="#11171C", width=2.5, dash="dot")))
    for m, g in d.groupby("model"):
        fig.add_trace(go.Scatter(x=g["ts"], y=g["point"].astype(float), name=m, line=dict(width=1.6)))
    return _base_layout(fig, f"{variate} · window {window} — every model's median forecast vs actual")


def viz_inspector(variate, model, window, view):
    p = preds(variate, model, window)
    if p.empty:
        return _empty("No predictions for this selection")
    p["ts"] = pd.to_datetime(p["ts"])
    for c in ["actual", "point"] + QCOLS:
        p[c] = p[c].astype(float)
    has_q = p["q10"].notna().any()
    fig = go.Figure()
    if view == "global":
        fs = full_series(variate); fs["ts"] = pd.to_datetime(fs["ts"])
        fig.add_trace(go.Scatter(x=fs["ts"], y=fs["value"], name="History", line=dict(color="#9AA7B2", width=1)))
        fig.add_vrect(x0=p["ts"].min(), x1=p["ts"].max(), fillcolor="#FFA600", opacity=0.12,
                      line_width=0, annotation_text="test window")
    if has_q:
        fig.add_trace(go.Scatter(x=p["ts"], y=p["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p["ts"], y=p["q10"], fill="tonexty", fillcolor=BAND.format(a=0.18),
                                 line=dict(width=0), name="Q10–Q90"))
        fig.add_trace(go.Scatter(x=p["ts"], y=p["q70"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p["ts"], y=p["q30"], fill="tonexty", fillcolor=BAND.format(a=0.30),
                                 line=dict(width=0), name="Q30–Q70"))
    fig.add_trace(go.Scatter(x=p["ts"], y=p["point"], name=f"{model} (median)", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=p["ts"], y=p["actual"], name="Actual", line=dict(color="#11171C", width=2, dash="dot")))
    return _base_layout(fig, f"{variate} · {model} · window {window} · {'quantile band' if has_q else 'point only'}")


# ---------------------------------------------------------------- step content
def step_features_table(variate):
    fr = feature_row(variate)
    if fr.empty:
        return html.Em("This variate was rejected by screening, so it has no features.")
    r = fr.iloc[0]
    rows = [("F1 Trend strength", f"{float(r['F1_trend_strength']):.3f}", int(r["F1_bit"])),
            ("F2 Trend linearity", f"{float(r['F2_trend_linearity']):.3f}", int(r["F2_bit"])),
            ("F3 Seasonality strength", f"{float(r['F3_seasonality_strength']):.3f}", int(r["F3_bit"])),
            ("F4 Seasonality corr", f"{float(r['F4_seasonality_correlation']):.3f}", int(r["F4_bit"])),
            ("F5 Residual ACF-1", f"{float(r['F5_residual_acf1']):.3f}", int(r["F5_bit"])),
            ("F6 Complexity", f"{float(r['F6_complexity']):.3f}", int(r["F6_bit"])),
            ("F7 Stationarity", str(int(r["F7_stationarity"])), int(r["F7_bit"]))]
    df = pd.DataFrame(rows, columns=["Feature", "Value", "Bit (> median)"])
    return html.Div([_table(df),
                     html.P(html.B(f"7-bit pattern code: {r['pattern_code']}"), className="mt-2")])


def metrics_table(variate):
    m = metrics_for(variate)
    if m.empty:
        return html.Em("No metrics for this variate.")
    m[["mase", "crps"]] = m[["mase", "crps"]].astype(float)
    agg = m.groupby("model")[["mase", "crps"]].mean().reset_index()
    base = agg[agg["model"] == "Seasonal-Naive"][["mase", "crps"]].iloc[0]
    agg["MASE (raw)"] = agg["mase"].round(3)
    agg["MASE (norm)"] = (agg["mase"] / base["mase"]).round(3)
    agg["CRPS (raw)"] = agg["crps"].round(3)
    agg["CRPS (norm)"] = (agg["crps"] / base["crps"]).round(3)
    out = agg[["model", "MASE (raw)", "MASE (norm)", "CRPS (raw)", "CRPS (norm)"]].sort_values("MASE (norm)")
    out = out.rename(columns={"model": "Model"})
    return _table(out)


# markdown explanation per step (rich + technical, MathJax)
EXPL = {
0: """### Why benchmark time-series foundation models?
**Time-series foundation models (TSFMs)** — Chronos, TimesFM, Moirai, Toto — are pretrained once and
applied **zero-shot**: forecast a series they were never fine-tuned on. *It's TIME* argues that the
benchmarks used to judge them have four structural problems:

1. **Legacy-constrained data / contamination** — benchmarks recycle old public datasets that have
   likely leaked into pretraining corpora, so "zero-shot" scores are inflated.
2. **Compromised data integrity** — little quality assurance; noisy, constant, or broken series slip in.
3. **Misaligned task formulation** — one fixed horizon (e.g. 720 steps) applied to every dataset,
   ignoring each series' operational decision cycle and physical predictability.
4. **Limited analysis perspective** — a single scalar (MASE) per domain hides *why* a model wins and can
   even reward a smooth forecast that misses real structure.

This app rebuilds the pipeline that fixes these — on a **synthetic set with known signatures** so every
step's verdict is checkable — and ends at the leaderboard on the right. Click **Next** to begin.""",

1: r"""### Step 1 — Forecastability screening (Algorithms 1 & 2)
Before modelling, each series is screened. A series is **kept** only if it passes every check; otherwise
it is dropped with a reason. The checks:

- **Type & length** — numeric and at least $3\times$ the seasonal period.
- **Missing rate** — $\rho_{miss}=\frac{|\{t: x_t=\text{NaN}\}|}{L}\le 0.30$; below that, gaps are
  forward/back-filled.
- **Constant / degenerate** — dropped if the top-5 values dominate ($\ge 0.5$ of observations) or the
  normalised entropy $< 0.1$.
- **White noise (Ljung-Box)** — if $\min p_{LB}(\text{lags }10,20) > 0.05$ the series is
  indistinguishable from noise → **unforecastable**, dropped.
- **Extreme outliers (sliding-window IQR, $k=9$)** — points beyond $k\cdot\text{IQR}$ of the local
  median are replaced; if too many, the series is dropped.

Pick a variate to see its raw series and verdict. Note that **`white_noise`** fails Ljung-Box and
**`near_constant`** fails the constant check — exactly as intended.""",

2: r"""### Step 2 — Structural features F1–F7 (STL)
Each surviving series is decomposed with **STL** (Seasonal-Trend via Loess): $x = T + S + R$. Seven
interpretable features summarise its shape:

- **F1 Trend strength** $= \max\left(0,\ 1-\frac{\mathrm{Var}(R)}{\mathrm{Var}(T+R)}\right)$
- **F2 Trend linearity** — linear coefficient of an orthogonal quadratic fit to $T$.
- **F3 Seasonality strength** $= \max\left(0,\ 1-\frac{\mathrm{Var}(R)}{\mathrm{Var}(S+R)}\right)$
- **F4 Seasonality correlation** — mean pairwise correlation between seasonal cycles.
- **F5 Residual ACF-1** — lag-1 autocorrelation left in $R$.
- **F6 Complexity** — spectral entropy of the raw series (high = noise-like).
- **F7 Stationarity** — ADF test indicator ($p<0.05 \Rightarrow 1$).

Each F1–F6 is binarised at the **population median** (bit = 1 if above), F7 is already binary → a
**7-bit pattern code** used to retrieve "series like this one" for pattern-level analysis. The panels
show the live STL split; the table shows this variate's features, bits, and code.""",

3: r"""### Step 3 — Forecasting task & horizons
Evaluation uses a **rolling-window** backtest: with horizon $H$ and stride $H$, the last part of each
series is split into $W$ non-overlapping windows; for each, the model sees only the history *before* the
window and forecasts $H$ steps ahead. The paper sets $H$ per dataset from its **operational decision
cycle** and **physical predictability** — human-driven series (traffic, demand) support longer horizons;
chaotic physics-driven ones (wind, some sensors) are capped short because accuracy degrades to noise.
The shaded regions are the $W$ test windows for the chosen variate.""",

4: r"""### Step 4 — The zero-shot model zoo
All models forecast **without fine-tuning**:

- **Seasonal-Naive** — repeats the last seasonal cycle; made *probabilistic* here by adding empirical
  quantiles of its in-sample seasonal residuals. This is the **baseline** everything is normalised to.
- **`ai_forecast()`** — Databricks' built-in SQL forecaster (point + interval); the "what does the
  platform give you out of the box" reference.
- **Chronos-Bolt** (`amazon/chronos-bolt-base`) — fast quantile TSFM.
- **Chronos-2** (`amazon/chronos-2`) — newer, multivariate-capable, quantile output.
- **TimesFM-2.5** (`google/timesfm-2.5-200m-pytorch`) — decoder-only TSFM with a continuous quantile head.

The chart overlays every model's **median** forecast against the actual for one window — same input, no
training. Notice they mostly track the truth while Seasonal-Naive lags.""",

5: r"""### Step 5 — Metrics: MASE & CRPS
Two complementary metrics, then normalisation:

- **MASE** (point) scales absolute error by the in-sample seasonal-naive error:
  $\text{MASE}=\frac{\frac{1}{H}\sum_i|Y_i-\hat Y_i|}{\frac{1}{n-s}\sum_{j>s}|Y_j-Y_{j-s}|}$.
- **CRPS** (probabilistic) $\approx$ the mean **weighted quantile loss** over 9 levels:
  $\text{wQL}[\alpha]=2\frac{\sum_i \Lambda_\alpha(\hat q_i(\alpha), Y_i)}{\sum_i |Y_i|}$,
  $\Lambda_\alpha(q,Y)=(\alpha-\mathbf{1}\{Y<q\})(Y-q)$.

Because raw scales differ across series, each model's metric is **divided by the Seasonal-Naive metric on
the same task** and aggregated with the **geometric mean** — so **< 1.0 beats the baseline**. The table
shows raw and normalised MASE/CRPS per model for the chosen variate.""",

6: r"""### Step 6 — The leaderboard
Aggregating normalised scores across all tasks with the geometric mean gives the overall ranking. On
this synthetic set **every modern model beats Seasonal-Naive** (< 1.0) — reproducing the paper's central
finding that recent TSFMs represent genuine capability, not benchmark overfitting. `ai_forecast` (the
platform built-in) lands right alongside the frontier TSFMs. The dashed line is the baseline.""",

7: r"""### Step 7 — Qualitative inspection (why scalars aren't enough)
A good MASE can still hide a forecast that misses the real structure (the paper shows a model scoring
well while flattening a spike). So *It's TIME* pairs metrics with **visual inspection**. This is the
forecast inspector: pick a **variate**, **model**, **window**, and **Global/Local** view. The shaded
bands are the **Q10–Q90** and **Q30–Q70** predictive quantiles; the dotted line is the actual. Compare a
strong TSFM against Seasonal-Naive on `chaotic_logistic` to see conservative "safe mean" behaviour, or on
`strong_seasonality` to see tight, well-calibrated bands.""",
}

STEP_TITLES = [
    "Why benchmark TSFMs", "Forecastability screening", "Structural features F1–F7",
    "Task & horizons", "The zero-shot model zoo", "Metrics: MASE & CRPS",
    "The leaderboard", "Qualitative inspection",
]
# which controls each step shows: v=variate m=model w=window r=view
STEP_CONTROLS = {0: "", 1: "v", 2: "v", 3: "v", 4: "vw", 5: "v", 6: "", 7: "vmwr"}
N = len(STEP_TITLES)

# ---------------------------------------------------------------- app + layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "It's TIME — Guided Study"
try:
    _variates, _models, _windows = dropdown_options()
except Exception:
    _variates, _models, _windows = [], [], []
_def_model = "Chronos-Bolt" if "Chronos-Bolt" in _models else (_models[0] if _models else None)


def _ctl_col(label, comp, cid):
    return dbc.Col([html.Label(label, className="fw-bold small"), comp], id=cid, md=3)


controls = dbc.Card(dbc.CardBody(dbc.Row([
    _ctl_col("Variate", dcc.Dropdown(_variates, _variates[0] if _variates else None, id="variate", clearable=False), "col-v"),
    _ctl_col("Model", dcc.Dropdown(_models, _def_model, id="model", clearable=False), "col-m"),
    _ctl_col("Window", dcc.Dropdown(_windows, _windows[-1] if _windows else None, id="window", clearable=False), "col-w"),
    _ctl_col("View", dcc.RadioItems(options=[{"label": " Local", "value": "local"}, {"label": " Global", "value": "global"}],
                                    value="local", id="view", inline=True,
                                    inputStyle={"marginRight": "4px", "marginLeft": "10px"}), "col-r"),
])), className="mb-3")

app.layout = dbc.Container([
    dcc.Store(id="step", data=0),
    html.H3("It's TIME — Guided Study", className="mt-3 mb-0"),
    html.P("A guided, technical walkthrough of the It's TIME (ICML 2026) forecasting-benchmark "
           "methodology — explanation + live data at every step.", className="text-muted"),
    dbc.Row([
        dbc.Col(html.Div(id="step-label", className="fw-bold"), md=8),
        dbc.Col(html.Div([
            dbc.Button("← Prev", id="prev", color="secondary", outline=True, size="sm", class_name="me-2"),
            dbc.Button("Next →", id="next", color="primary", size="sm"),
        ], className="text-end"), md=4),
    ], className="align-items-center mb-1"),
    dbc.Progress(id="progress", value=0, style={"height": "6px"}, class_name="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Markdown(id="explanation", mathjax=True, dangerously_allow_html=False,
                             style={"fontSize": "0.95rem"}), md=5),
        dbc.Col([controls, dcc.Loading(html.Div(id="visual"))], md=7),
    ]),
], fluid=True)


# ---------------------------------------------------------------- callbacks
@app.callback(Output("step", "data"),
              Input("prev", "n_clicks"), Input("next", "n_clicks"), State("step", "data"),
              prevent_initial_call=True)
def navigate(prev_c, next_c, step):
    trig = dash.callback_context.triggered[0]["prop_id"]
    step = step or 0
    if trig.startswith("next"):
        step = min(N - 1, step + 1)
    elif trig.startswith("prev"):
        step = max(0, step - 1)
    return step


@app.callback(
    Output("step-label", "children"), Output("progress", "value"),
    Output("explanation", "children"), Output("visual", "children"),
    Output("col-v", "style"), Output("col-m", "style"), Output("col-w", "style"), Output("col-r", "style"),
    Output("prev", "disabled"), Output("next", "disabled"),
    Input("step", "data"), Input("variate", "value"), Input("model", "value"),
    Input("window", "value"), Input("view", "value"))
def render(step, variate, model, window, view):
    step = step or 0
    shown = STEP_CONTROLS.get(step, "")
    hide = {"display": "none"}
    styles = [None if "v" in shown else hide, None if "m" in shown else hide,
              None if "w" in shown else hide, None if "r" in shown else hide]

    try:
        if step == 0:
            vis = _graph(viz_leaderboard())
        elif step == 1:
            vis = html.Div([_graph(viz_screening(variate), "34vh"),
                            html.Div(_table(quality_df()[["variate", "predictable", "reason", "n_obs"]]),
                                     style={"maxHeight": "22vh", "overflowY": "auto"})])
        elif step == 2:
            vis = html.Div([_graph(viz_stl(variate), "40vh"), step_features_table(variate)])
        elif step == 3:
            vis = _graph(viz_windows(variate))
        elif step == 4:
            vis = _graph(viz_overlay(variate, window))
        elif step == 5:
            vis = metrics_table(variate)
        elif step == 6:
            vis = _graph(viz_leaderboard())
        else:
            vis = _graph(viz_inspector(variate, model, window, view), "56vh")
    except Exception as e:  # graceful per-step fallback
        vis = _graph(_empty(f"Could not render this step: {str(e)[:200]}"))

    label = f"Step {step + 1} of {N} — {STEP_TITLES[step]}"
    return (label, int((step + 1) / N * 100), EXPL[step], vis,
            styles[0], styles[1], styles[2], styles[3], step == 0, step == N - 1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("DATABRICKS_APP_PORT", "8000")), debug=False)
