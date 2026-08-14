from __future__ import annotations

from datetime import date
import logging
import traceback

import pandas as pd

from funding_dashboard.http import HttpClient
from funding_dashboard.settings import Settings
from funding_dashboard.storage import Storage
from funding_dashboard.sources.nyfed import fetch_reference_rates
from funding_dashboard.sources.fred import fetch_many
from funding_dashboard.sources.ofr import fetch_dataset
from funding_dashboard.sources.treasury import fetch_auctions
from funding_dashboard.validation import validate_auctions, validate_observations

LOGGER = logging.getLogger(__name__)


class PipelineUpdateError(RuntimeError):
    def __init__(self, message: str, results: list[dict]):
        super().__init__(message)
        self.results = results


def _incremental_start(storage: Storage, source: str, dataset: str, fallback: str, lookback_days: int) -> str:
    max_date = storage.max_date(source, dataset)
    if max_date is None:
        return fallback
    return (max_date - pd.Timedelta(days=lookback_days)).date().isoformat()


def run_update(settings: Settings) -> dict:
    cfg = settings.config
    client = HttpClient(settings.cache_dir, cfg["http"])
    results: list[dict] = []
    fallback = settings.start_date
    lookback = int(cfg["project"].get("lookback_days", 45))
    today = date.today().isoformat()
    validation_cfg = cfg.get("validation", {})

    with Storage(settings.db_path) as storage:
        def execute(source: str, dataset: str, fn, *, required: bool = True) -> None:
            result = {
                "source": source,
                "dataset": dataset,
                "required": required,
                "status": "error",
                "rows_fetched": 0,
                "rows_loaded": 0,
                "validation": None,
                "error": None,
            }
            try:
                frame = fn()
                result["rows_fetched"] = len(frame)
                report = validate_observations(frame, validation_cfg, name=f"{source}:{dataset}")
                result["validation"] = report.to_dict()
                report.raise_for_errors()
                rows = storage.upsert_observations(frame)
                result["rows_loaded"] = rows
                result["status"] = "ok"
                storage.log_run(source, dataset, "ok", rows)
                LOGGER.info("Source completed source=%s dataset=%s rows=%s", source, dataset, rows)
            except Exception as exc:
                LOGGER.exception("%s/%s update failed", source, dataset)
                message = f"{exc}\n{traceback.format_exc()}"
                storage.log_run(source, dataset, "error", 0, message)
                result["error"] = str(exc)
            results.append(result)

        if cfg["sources"]["nyfed"].get("enabled", True):
            scfg = cfg["sources"]["nyfed"]
            start = _incremental_start(storage, "nyfed", "reference_rates", fallback, lookback)
            execute(
                "nyfed",
                "reference_rates",
                lambda: fetch_reference_rates(
                    client,
                    start,
                    today,
                    scfg.get("max_days_per_request", 365),
                ),
                required=scfg.get("required", True),
            )

        if cfg["sources"]["fred"].get("enabled", True):
            scfg = cfg["sources"]["fred"]
            fred_series = scfg["series"]
            fred_starts: dict[str, str] = {}
            for series_id in fred_series:
                max_date = storage.max_date("fred", "fred", series_id=series_id)
                fred_starts[series_id] = (
                    fallback
                    if max_date is None
                    else (max_date - pd.Timedelta(days=lookback)).date().isoformat()
                )
            execute(
                "fred",
                "fred",
                lambda: fetch_many(
                    client,
                    fred_series,
                    fred_starts,
                    strict=scfg.get("fail_on_any_series_error", True),
                ),
                required=scfg.get("required", True),
            )

        if cfg["sources"]["ofr"].get("enabled", True):
            scfg = cfg["sources"]["ofr"]
            for dataset in scfg.get("datasets", []):
                start = _incremental_start(storage, "ofr", dataset, fallback, max(35, lookback))
                execute(
                    "ofr",
                    dataset,
                    lambda d=dataset, s=start: fetch_dataset(client, d, s, hedge_fund=False),
                    required=scfg.get("required", True),
                )
            for dataset in scfg.get("hedge_fund_datasets", []):
                start = _incremental_start(storage, "ofr_hf", dataset, fallback, max(35, lookback))
                execute(
                    "ofr_hf",
                    dataset,
                    lambda d=dataset, s=start: fetch_dataset(client, d, s, hedge_fund=True),
                    required=scfg.get("required", True),
                )

        if cfg["sources"]["treasury"].get("enabled", True):
            scfg = cfg["sources"]["treasury"]
            result = {
                "source": "treasury",
                "dataset": "auctions",
                "required": scfg.get("required", True),
                "status": "error",
                "rows_fetched": 0,
                "rows_loaded": 0,
                "validation": None,
                "error": None,
            }
            try:
                frame = fetch_auctions(
                    client,
                    scfg.get("days_back", 90),
                    scfg.get("days_forward", 45),
                )
                result["rows_fetched"] = len(frame)
                report = validate_auctions(frame)
                result["validation"] = report.to_dict()
                report.raise_for_errors()
                rows = storage.upsert_auctions(frame)
                result["rows_loaded"] = rows
                result["status"] = "ok"
                storage.log_run("treasury", "auctions", "ok", rows)
            except Exception as exc:
                LOGGER.exception("Treasury auction update failed")
                storage.log_run("treasury", "auctions", "error", 0, str(exc))
                result["error"] = str(exc)
            results.append(result)

    required_failures = [r for r in results if r["required"] and r["status"] != "ok"]
    summary = {
        "source_results": results,
        "rows_loaded": sum(int(r["rows_loaded"]) for r in results),
        "failed_sources": len([r for r in results if r["status"] != "ok"]),
        "required_failures": len(required_failures),
    }
    if required_failures:
        names = ", ".join(f"{r['source']}:{r['dataset']}" for r in required_failures)
        raise PipelineUpdateError(f"Required data sources failed: {names}", results)
    return summary
