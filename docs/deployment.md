# Deployment runbook

All commands assume the Databricks CLI with a profile pointed at the target workspace
(`--profile <PROFILE>`; the reference build used `wscalioni-classic`). Replace the catalog, schema,
warehouse id, workspace user path, and GPU cluster id with your own.

## Prerequisites

- Unity Catalog **catalog that already exists** (the notebooks only `CREATE SCHEMA` — see gotcha below).
- A **serverless SQL warehouse** (for `ai_forecast` + dashboard/app). Note its id.
- For the two GPU models: a small **GPU cluster** (single-node `g4dn.xlarge` / T4 16 GB is enough).
- Databricks CLI ≥ 0.292; authenticated (`databricks auth profiles`).

## 1. Deploy the notebooks

```bash
WSDIR="/Users/<you>/its-time-benchmark"
databricks workspace mkdirs "$WSDIR" --profile <PROFILE>
for nb in 01_screening_and_features 02_zeroshot_bakeoff; do
  databricks workspace import "$WSDIR/$nb" --file $nb.py \
    --format SOURCE --language PYTHON --overwrite --profile <PROFILE>
done
```

## 2. Run the pipeline

Notebook 1 (serverless) — widgets `series_len`, `season_period`, `source` (`synthetic`|`table`):

```bash
databricks jobs submit --profile <PROFILE> --json '{
  "run_name": "its-time-slice1",
  "tasks": [{"task_key": "screen", "notebook_task": {
    "notebook_path": "'$WSDIR'/01_screening_and_features",
    "base_parameters": {"series_len": "800", "season_period": "24"}}}]}'
```

Notebook 2 — on **serverless** (Seasonal-Naive + ai_forecast + Chronos-Bolt):

```bash
databricks jobs submit --profile <PROFILE> --json '{
  "run_name": "its-time-slice2",
  "tasks": [{"task_key": "bakeoff", "notebook_task": {
    "notebook_path": "'$WSDIR'/02_zeroshot_bakeoff",
    "base_parameters": {"horizon": "24", "n_windows": "3", "season_period": "24",
                        "frequency": "1 hour", "warehouse_id": "<WAREHOUSE_ID>"}}}]}'
```

…or on the **GPU cluster** to add Chronos-2 + TimesFM-2.5 (pin `existing_cluster_id`, set `gpu_models`):

```bash
databricks clusters start <GPU_CLUSTER_ID> --profile <PROFILE>   # wait for RUNNING
databricks jobs submit --profile <PROFILE> --json '{
  "run_name": "its-time-slice2-gpu",
  "tasks": [{"task_key": "bakeoff_gpu", "existing_cluster_id": "<GPU_CLUSTER_ID>",
    "notebook_task": {"notebook_path": "'$WSDIR'/02_zeroshot_bakeoff",
      "base_parameters": {"horizon": "24", "n_windows": "3", "season_period": "24",
                          "frequency": "1 hour", "warehouse_id": "<WAREHOUSE_ID>",
                          "gpu_models": "Chronos-2,TimesFM-2.5"}}}]}'
```

Terminate the GPU cluster when done to avoid idle cost.

## 3. Deploy the AI/BI dashboard

Queries use bare table names; catalog/schema are supplied by flags at create time.

```bash
databricks lakeview create \
  --display-name "It's TIME — Forecasting Leaderboard" \
  --warehouse-id "<WAREHOUSE_ID>" \
  --dataset-catalog "<CATALOG>" --dataset-schema "<SCHEMA>" \
  --serialized-dashboard "$(cat dashboard/its_time.json)" \
  --json '{"parent_path": "'$WSDIR'"}' --profile <PROFILE>
databricks lakeview publish <DASHBOARD_ID> --warehouse-id "<WAREHOUSE_ID>" --profile <PROFILE>
```

## 4. Deploy the guided-study app

```bash
# create app with the SQL warehouse attached as a resource
databricks apps create --profile <PROFILE> --json '{
  "name": "its-time-inspector",
  "resources": [{"name": "sql-warehouse",
                 "sql_warehouse": {"id": "<WAREHOUSE_ID>", "permission": "CAN_USE"}}]}'

# grant the app service principal read access (SP client id from `databricks apps get`)
# GRANT USE CATALOG / USE SCHEMA / SELECT ON SCHEMA <CATALOG>.<SCHEMA> TO `<SP_CLIENT_ID>`

# upload + deploy
APPDIR="$WSDIR/app-src"
databricks workspace import-dir app "$APPDIR" --overwrite --profile <PROFILE>
databricks apps deploy its-time-inspector --source-code-path "$APPDIR" --profile <PROFILE>
databricks apps get its-time-inspector --profile <PROFILE>   # expect deploy SUCCEEDED + URL
```

The app reads catalog/schema/warehouse from `app/app.yaml` (`valueFrom: sql-warehouse` +
`ITS_TIME_CATALOG` / `ITS_TIME_SCHEMA`). It is OAuth-gated; open the URL as an authorised user.

## Environment gotchas found during the build

- **`CREATE CATALOG` is blocked** by the account default-storage rule — the catalog must pre-exist; the
  notebooks only `CREATE SCHEMA`.
- **`ai_forecast` is preview-disabled on serverless notebook (Spark Connect) compute** but enabled on SQL
  warehouses — Notebook 2 routes that call through the warehouse via `databricks-sql-connector`.
- **`ai_forecast` horizon is endpoint-exclusive** — request a couple of extra steps and slice to `H`.
- **Chronos-Bolt** `predict_quantiles` takes the series as the positional `inputs` arg (not `context=`).
- **Chronos-2** uses `Chronos2Pipeline.predict_df(...)` with quantile columns named `'0.1'`…`'0.9'`.
- **TimesFM-2.5** uses `TimesFM_2p5_200M_torch` → `.compile(ForecastConfig(...))` → `forecast()`
  returning `(point (B,H), quantiles (B,H,10))` where col 0 is the mean and cols 1–9 are Q0.1–Q0.9.
- **App layout**: keep the Plotly title at the top and the legend at the bottom (`y=-0.12`) so they don't
  overlap in the narrow app pane.
