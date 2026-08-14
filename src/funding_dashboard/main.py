from __future__ import annotations

import argparse
from datetime import date, timedelta
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

from funding_dashboard.analytics import run_analytics
from funding_dashboard.pipeline import PipelineUpdateError, run_update
from funding_dashboard.publication import (
    atomic_publish_run,
    planned_s3_uris,
    publish_outputs_to_s3,
    restore_latest_database_checkpoint,
    write_validation_report,
)
from funding_dashboard.run_metadata import utc_now, write_run_metadata
from funding_dashboard.settings import load_settings
from funding_dashboard.storage import Storage
from funding_dashboard.validation import assess_staleness

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _new_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _copy_existing_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, target)
        LOGGER.info("Copied existing database into staging path=%s", target)


def _populate_smoke_database(db_path: Path) -> int:
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
    rows: list[dict] = []
    series = {
        "SOFR_RATE": (5.30, "%", "nyfed"),
        "SOFR_P25": (5.28, "%", "nyfed"),
        "SOFR_P75": (5.32, "%", "nyfed"),
        "SOFR_P99": (5.38, "%", "nyfed"),
        "SOFR_VOLUME": (1850.0, "USD bn", "nyfed"),
        "TGCR_RATE": (5.27, "%", "nyfed"),
        "TGCR_P25": (5.25, "%", "nyfed"),
        "TGCR_P75": (5.29, "%", "nyfed"),
        "EFFR_RATE": (5.29, "%", "nyfed"),
        "EFFR_VOLUME": (95.0, "USD bn", "nyfed"),
        "OBFR_RATE": (5.30, "%", "nyfed"),
        "BGCR_RATE": (5.28, "%", "nyfed"),
        "IORB": (5.40, "%", "fred"),
        "WRESBAL": (3200.0, "USD bn", "fred"),
        "WTREGEN": (760.0, "USD bn", "fred"),
        "RRPONTSYD": (310.0, "USD bn", "fred"),
        "WALCL": (7300.0, "USD bn", "fred"),
        "DGS1MO": (5.15, "%", "fred"),
        "DGS3MO": (5.10, "%", "fred"),
        "DCPF1M": (5.35, "%", "fred"),
        "DCPF3M": (5.40, "%", "fred"),
        "DCPN30": (5.30, "%", "fred"),
        "DCPN3M": (5.35, "%", "fred"),
        "RIFSPPNA2P2D30NB": (5.65, "%", "fred"),
        "RIFSPPNA2P2D90NB": (5.75, "%", "fred"),
        "FINCP": (1150.0, "USD bn", "fred"),
        "NFINCP": (980.0, "USD bn", "fred"),
        "LTDACBW027NBOG": (2150.0, "USD bn", "fred"),
    }
    for i, dt in enumerate(dates):
        wave = np.sin(i / 23.0)
        for sid, (base, unit, source) in series.items():
            scale = 0.015 if unit == "%" else max(abs(base) * 0.002, 0.1)
            value = base + scale * wave + scale * i / len(dates)
            rows.append({
                "source": source,
                "dataset": "reference_rates" if source == "nyfed" else "fred",
                "series_id": sid,
                "series_name": sid,
                "date": dt,
                "value": float(value),
                "unit": unit,
                "metadata_json": '{"synthetic": true}',
            })

    auctions = pd.DataFrame([{
        "record_date": date.today(),
        "auction_date": date.today(),
        "issue_date": date.today(),
        "maturity_date": date.today() + timedelta(days=28),
        "cusip": "SMOKETEST",
        "security_type": "Bill",
        "security_term": "4-Week",
        "offering_amt": 75000000000,
        "total_accepted": 75000000000,
        "bid_to_cover_ratio": 2.85,
        "high_yield": 5.10,
        "raw_json": '{"synthetic": true}',
    }])

    with Storage(db_path) as storage:
        count = storage.upsert_observations(pd.DataFrame(rows))
        storage.upsert_auctions(auctions)
    return count


def execute(config_path: str, mode: str) -> dict:
    _configure_logging()
    started_at = utc_now()
    started_clock = time.monotonic()
    run_id = _new_run_id()
    run_date = started_at.date().isoformat()
    settings = load_settings(config_path)
    run_root = settings.staging_dir / f"run_id={run_id}"
    staged_db = run_root / "funding.duckdb"
    staged_outputs = run_root / "outputs"
    publication_output_dir = settings.output_dir
    publication_database_path: Path | None = None
    run_root.mkdir(parents=True, exist_ok=False)

    metadata: dict = {
        "run_id": run_id,
        "run_date": run_date,
        "run_env": os.getenv("RUN_ENV", "local"),
        "mode": mode,
        "output_mode": settings.output_mode,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "exit_code": None,
        "database_path": str(settings.db_path),
        "output_dir": str(settings.output_dir),
        "source_results": [],
        "rows_loaded": 0,
        "stale_series_count": 0,
        "critical_stale_series_count": 0,
        "validation_reports": [],
        "output_files": [],
        "s3_objects": [],
        "database_checkpoint_restored": False,
        "database_checkpoint_run_id": None,
        "error": None,
    }
    metadata_path: Path | None = None

    try:
        settings.validate_runtime()

        if mode == "full":
            if settings.db_path.exists():
                _copy_existing_database(settings.db_path, staged_db)
            elif settings.output_mode in {"s3", "both"}:
                checkpoint = restore_latest_database_checkpoint(settings, staged_db)
                if checkpoint is not None:
                    metadata["database_checkpoint_restored"] = True
                    metadata["database_checkpoint_run_id"] = checkpoint.get("run_id")
            run_settings = settings.with_project_paths(database=staged_db, output_dir=staged_outputs)
            update_summary = run_update(run_settings)
            metadata["source_results"] = update_summary["source_results"]
            metadata["rows_loaded"] = update_summary["rows_loaded"]
            publish_db = True
        elif mode == "analytics-only":
            if not settings.db_path.exists():
                raise FileNotFoundError(f"Database not found: {settings.db_path}")
            run_settings = settings.with_project_paths(database=settings.db_path, output_dir=staged_outputs)
            publish_db = False
        elif mode == "smoke-test":
            run_settings = settings.with_project_paths(database=staged_db, output_dir=staged_outputs)
            metadata["rows_loaded"] = _populate_smoke_database(staged_db)
            publish_db = True
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        with Storage(run_settings.db_path) as storage:
            observations = storage.observations()
        stale_report = assess_staleness(
            observations,
            settings.config.get("validation", {}),
            as_of=date.today(),
        )
        metadata["validation_reports"].append(stale_report.to_dict())
        metadata["stale_series_count"] = len(stale_report.stale_series)
        metadata["critical_stale_series_count"] = len(
            [item for item in stale_report.stale_series if item["critical"]]
        )
        stale_report.raise_for_errors()

        analytics = run_analytics(run_settings)
        write_validation_report(staged_outputs, metadata["validation_reports"])
        metadata["output_files"] = sorted(path.name for path in staged_outputs.iterdir() if path.is_file())

        if settings.output_mode == "s3":
            # Fargate storage is ephemeral. Publishing directly from the staged
            # directory avoids an unnecessary overlay-filesystem rename and
            # keeps the staged files available until the S3 commit completes.
            publication_output_dir = staged_outputs
            if mode in {"full", "analytics-only"}:
                publication_database_path = run_settings.db_path
        else:
            atomic_publish_run(
                staged_db=staged_db if publish_db else None,
                final_db=settings.db_path if publish_db else None,
                staged_outputs=staged_outputs,
                final_outputs=settings.output_dir,
            )
            if settings.output_mode == "both":
                if mode == "full":
                    publication_database_path = settings.db_path
                elif mode == "analytics-only":
                    publication_database_path = run_settings.db_path

        metadata["status"] = "success"
        metadata["exit_code"] = 0
        latest_scores = analytics["scores"].dropna(subset=["liquidity_risk_score"])
        if not latest_scores.empty:
            latest = latest_scores.iloc[-1]
            metadata["latest_liquidity_risk_score"] = float(latest["liquidity_risk_score"])
            metadata["latest_regime"] = str(latest["regime"])

    except PipelineUpdateError as exc:
        metadata["source_results"] = exc.results
        metadata["rows_loaded"] = sum(int(r.get("rows_loaded", 0)) for r in exc.results)
        metadata["status"] = "failure"
        metadata["exit_code"] = 1
        metadata["error"] = str(exc)
        LOGGER.exception("Pipeline update failed")
    except Exception as exc:
        metadata["status"] = "failure"
        metadata["exit_code"] = 1
        metadata["error"] = str(exc)
        LOGGER.exception("Pipeline run failed")
    finally:
        finished_at = utc_now()
        metadata["finished_at_utc"] = finished_at.isoformat()
        metadata["duration_seconds"] = round(time.monotonic() - started_clock, 3)
        if metadata["status"] == "success" and settings.output_mode in {"s3", "both"}:
            try:
                metadata["s3_objects"] = planned_s3_uris(
                    settings,
                    publication_output_dir,
                    run_id,
                    run_date,
                    publication_database_path,
                )
                metadata_path = write_run_metadata(metadata, settings.run_metadata_dir)
                metadata["s3_objects"] = publish_outputs_to_s3(
                    settings,
                    publication_output_dir,
                    metadata_path,
                    run_id,
                    run_date,
                    publication_database_path,
                )
            except Exception as exc:
                metadata["status"] = "failure"
                metadata["exit_code"] = 1
                metadata["s3_objects"] = []
                metadata["error"] = f"S3 publication failed: {exc}"
                LOGGER.exception("S3 publication failed")

        metadata_path = write_run_metadata(metadata, settings.run_metadata_dir)
        LOGGER.info(
            "RUN_SUMMARY run_id=%s status=%s rows_loaded=%s stale_series=%s "
            "duration_seconds=%s exit_code=%s",
            run_id,
            metadata["status"],
            metadata["rows_loaded"],
            metadata["stale_series_count"],
            metadata["duration_seconds"],
            metadata["exit_code"],
        )

        if run_root.exists():
            shutil.rmtree(run_root, ignore_errors=True)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U.S. funding and liquidity batch pipeline")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--mode",
        choices=["full", "analytics-only", "smoke-test"],
        default=os.getenv("RUN_MODE", "full"),
    )
    return parser


def cli() -> None:
    args = build_parser().parse_args()
    metadata = execute(args.config, args.mode)
    raise SystemExit(int(metadata["exit_code"]))


if __name__ == "__main__":
    cli()
