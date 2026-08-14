# U.S. Funding & Liquidity Dashboard v2

This version expands the original SOFR–IORB / TGA / reserves monitor into a broader money-market dashboard.


## Docker batch execution (AWS-ready)

The batch runner now supports three modes:

- `smoke-test`: synthetic, offline end-to-end verification
- `analytics-only`: regenerate outputs from an existing DuckDB without API calls
- `full`: API ingestion, validation, analytics, atomic local publication, optional S3 publication

Build and run the offline smoke test from PowerShell:

```powershell
docker build --tag funding-liquidity-pipeline:local .
New-Item -ItemType Directory -Force .\docker_data | Out-Null
docker compose run --rm smoke-test
```

See `LOCAL_DOCKER_RUNBOOK.md` for the full Windows workflow, output checks, live-API test, exit-code verification, and troubleshooting.

## What was added

- New York Fed: **EFFR, OBFR, TGCR, BGCR, SOFR**, including rates, volumes, percentiles and policy target range.
- Derived spreads: EFFR–IORB, OBFR–IORB, OBFR–EFFR, TGCR–IORB, BGCR–IORB, SOFR–IORB, SOFR–TGCR, BGCR–TGCR and SOFR–EFFR.
- OFR STFM: DVP, GCF, tri-party and term repo indicators; MMF indicators; primary-dealer repo/reverse-repo and settlement fails.
- OFR Hedge Fund Monitor: sponsored repo and reverse-repo volumes.
- Federal Reserve/FRED: reserves, TGA, ON RRP, Fed assets, CP rates/outstandings, A2/P2 spreads, large time deposits and short Treasury rates.
- Treasury Fiscal Data: recent and upcoming bill/note/bond auctions and settlement dates.
- Component-based liquidity risk score, regime classification, 1-week/1-month/3-month relative-value table, and rule-based portfolio actions.

## Data-access design

The project uses official public APIs and download endpoints only. It does **not** bypass CAPTCHAs, robots rules, authentication, throttling, or anti-bot controls.

Efficiency and site protection:

- SQLite-backed HTTP cache (`requests-cache`).
- Incremental refresh with a configurable overlap window.
- New York Fed history is split into one-year chunks.
- OFR datasets are requested once per refresh, not one request per series.
- Exponential retry/backoff and 429 handling.
- Small delay between requests.
- Weekly/monthly series are forward-filled only for limited periods and receive a data-coverage flag.

## Install on Windows

```powershell
cd "C:\path\to\funding_dashboard_v2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
copy .env.example .env
```

Edit `.env` and replace the SEC user-agent email with your real contact information.

## First data load

```powershell
python run_update.py
```

The first load starts in April 2018. OFR repo/MMF datasets can be large, so the initial run takes longer than later incremental updates.

## Start dashboard

```powershell
$env:DATABASE_PATH = "C:\funding_dashboard_runtime\funding.duckdb"
python -m streamlit run run_dashboard.py
```

Docker Desktop uses the same runtime database through the bind mount in
`compose.yaml`:

```powershell
docker compose build dashboard
docker compose up -d dashboard
docker compose logs -f dashboard
```

Open `http://localhost:8501`. Run `docker compose run --rm full` separately
when source data need to be refreshed; the dashboard process does not ingest data.

## Streamlit Community Cloud (private S3 Parquet reader)

The dashboard can run without DuckDB by reading the immutable artifacts named by
S3 `latest.json`. This mode uses `funding_data.parquet`, `liquidity_metrics.parquet`,
the series catalog, and the small report files; it never lists an S3 prefix or
downloads the DuckDB checkpoint. Local execution remains DuckDB-based by default.

For a local S3-reader check:

```powershell
$env:DASHBOARD_DATA_SOURCE = "s3"
$env:S3_BUCKET = "funding-liquidity-data-502055890072-us-west-2"
$env:AWS_REGION = "us-west-2"
python -m streamlit run run_dashboard.py
```

For Streamlit Community Cloud:

1. Push this project to a private GitHub repository and create an app with `run_dashboard.py` as the entrypoint.
2. Attach the read-only policy in `aws/streamlit-s3-read-policy.json` to a dedicated IAM user. Do not reuse the ECS deployment user or task role.
3. Create an access key for that read-only user and copy `.streamlit/secrets.toml.example` into the app's **Secrets** settings, replacing every placeholder.
4. Keep the S3 bucket private. The dashboard needs only `s3:GetObject`; it does not need `s3:ListBucket`, write access, or the DuckDB checkpoint.

Community Cloud cannot use the ECS task role directly, so its read-only AWS key is
stored as a Streamlit secret. Rotate that key if it is exposed. The data refresh
continues to run on ECS; the dashboard refreshes its cached S3 snapshot every 15 minutes.

## Discover OFR labels

OFR series names can evolve. The code stores all available series, then uses rules in `config.yml` to choose practical proxies. Review matches in the dashboard's **Data Quality** tab.

```powershell
python scripts/discover_ofr_series.py --dataset repo --query "DVP"
python scripts/discover_ofr_series.py --dataset mmf --query "Weighted Average"
python scripts/discover_ofr_series.py --dataset nypd --query "Fails to Deliver"
python scripts/discover_ofr_series.py --dataset ficc --query "Sponsored"
```

If a rule is not matched, edit `ofr_rules` in `config.yml`; no source-code change is required.

## Schedule daily updates

### Windows Task Scheduler

Program/script:

```text
C:\path\to\funding_dashboard_v2\.venv\Scripts\python.exe
```

Arguments:

```text
C:\path\to\funding_dashboard_v2\run_update.py
```

Start in:

```text
C:\path\to\funding_dashboard_v2
```

Run once each U.S. business day after 10:00 a.m. ET. New York Fed reference rates are published in the morning, while OFR repo preliminary data arrive later; a second run after 4:00 p.m. ET is optional because the cache and incremental logic prevent wasteful reloads.

## Outputs

- `data/funding.duckdb`: normalized data store
- `data/outputs/features.parquet`: derived indicators
- `data/outputs/scores.parquet`: risk score and components
- `data/outputs/catalog.parquet`: source-series coverage and observation dates
- `data/outputs/relative_value.csv`: 1w/1m/3m comparison
- `data/outputs/treasury_auctions.csv`: auction calendar
- `data/outputs/recommendations.json`: portfolio actions
- `data/outputs/selected_ofr_series.json`: exact OFR series selected by each rule

For S3 publication, every artifact is written to an immutable `run_id` path.
After all artifacts and the run-status commit marker are verified, `latest.json`
is conditionally updated so an older overlapping run cannot replace a newer one.
Full and analytics-only S3 runs also publish a checksummed DuckDB checkpoint;
the next full run restores it before performing the incremental refresh.
The ECS task role needs the object-level permissions in
`aws/ecs-task-s3-policy.json` for checkpoint restore and publication.

## Liquidity-risk score

The score combines fixed, economically interpretable thresholds (70%) with a
five-year rolling percentile anomaly (30%). Fixed thresholds use policy-rate
spreads, secured-rate dispersion, bank-asset-scaled reserve and emergency
facility measures, and CP/CD spreads and changes. The dashboard also reports the
highest component score so a concentrated stress event is not hidden by the
weighted total.

Regime cutoffs are 25, 40, 55, and 70 for Ample, Normal, Watch, Stressed, and
Severe. Thresholds and weights are defined in `config.yml`.

Run the reproducible event backtest against an existing database:

```powershell
python scripts/backtest_liquidity_score.py `
  --database "C:\funding_dashboard_runtime\funding.duckdb" `
  --output-dir data\backtest
```

Reference-rate underlying volumes come directly from the New York Fed API. The
OFR FNYR underlying-volume mnemonics are used as an official historical fallback
and converted from dollars to USD billions.

## Important interpretation notes

- Higher repo volume is not automatically stress. The score combines volume with spreads, dispersion and dealer fails.
- The MMF component is an OFR aggregate allocation proxy based on total investments, repo share, and bank-related asset share. It is not a substitute for SEC Form N-MFP fund-type, WAM/WAL, or daily/weekly liquidity fields.
- Large time deposits and MMF CD allocations are **CD proxies**, not a live institutional CD yield curve.
- `Fed assets – TGA – ON RRP` is included as a context indicator, not treated as a sufficient liquidity model.
- Portfolio recommendations are research rules, not fund-specific investment advice.
