from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import duckdb
import pandas as pd


OBS_COLUMNS = [
    "source", "dataset", "series_id", "series_name", "date", "value", "unit", "metadata_json"
]


class Storage:
    def __init__(self, db_path: Path, read_only: bool = False):
        if not read_only:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path), read_only=read_only)
        if not read_only:
            self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                source VARCHAR,
                dataset VARCHAR,
                series_id VARCHAR,
                series_name VARCHAR,
                date DATE,
                value DOUBLE,
                unit VARCHAR,
                metadata_json VARCHAR,
                ingested_at TIMESTAMP,
                PRIMARY KEY (source, dataset, series_id, date)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS treasury_auctions (
                record_date DATE,
                auction_date DATE,
                issue_date DATE,
                maturity_date DATE,
                cusip VARCHAR,
                security_type VARCHAR,
                security_term VARCHAR,
                offering_amt DOUBLE,
                total_accepted DOUBLE,
                bid_to_cover_ratio DOUBLE,
                high_yield DOUBLE,
                raw_json VARCHAR,
                ingested_at TIMESTAMP,
                PRIMARY KEY (cusip, auction_date)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS source_runs (
                source VARCHAR,
                dataset VARCHAR,
                run_at TIMESTAMP,
                status VARCHAR,
                rows_loaded BIGINT,
                message VARCHAR
            )
        """)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def max_date(self, source: str, dataset: str, series_id: str | None = None) -> pd.Timestamp | None:
        if series_id is None:
            sql = "SELECT MAX(date) FROM observations WHERE source=? AND dataset=?"
            params = [source, dataset]
        else:
            sql = "SELECT MAX(date) FROM observations WHERE source=? AND dataset=? AND series_id=?"
            params = [source, dataset, series_id]
        row = self.conn.execute(sql, params).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def upsert_observations(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        work = df.copy()
        for col in OBS_COLUMNS:
            if col not in work.columns:
                work[col] = None
        work = work[OBS_COLUMNS]
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        work = work.dropna(subset=["source", "dataset", "series_id", "date", "value"])
        work["metadata_json"] = work["metadata_json"].fillna("{}")
        work["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
        self.conn.register("incoming_obs", work)
        self.conn.execute("""
            INSERT OR REPLACE INTO observations
            SELECT source, dataset, series_id, series_name, date, value, unit,
                   metadata_json, ingested_at
            FROM incoming_obs
        """)
        self.conn.unregister("incoming_obs")
        return len(work)

    def upsert_auctions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        work = df.copy()
        for c in ["record_date", "auction_date", "issue_date", "maturity_date"]:
            work[c] = pd.to_datetime(work[c], errors="coerce").dt.date
        for c in ["offering_amt", "total_accepted", "bid_to_cover_ratio", "high_yield"]:
            work[c] = pd.to_numeric(work.get(c), errors="coerce")
        work["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
        cols = [
            "record_date", "auction_date", "issue_date", "maturity_date", "cusip",
            "security_type", "security_term", "offering_amt", "total_accepted",
            "bid_to_cover_ratio", "high_yield", "raw_json", "ingested_at"
        ]
        self.conn.register("incoming_auc", work[cols])
        self.conn.execute("INSERT OR REPLACE INTO treasury_auctions SELECT * FROM incoming_auc")
        self.conn.unregister("incoming_auc")
        return len(work)

    def log_run(self, source: str, dataset: str, status: str, rows: int, message: str = "") -> None:
        self.conn.execute(
            "INSERT INTO source_runs VALUES (?, ?, ?, ?, ?, ?)",
            [source, dataset, datetime.now(timezone.utc).replace(tzinfo=None), status, rows, message[:2000]],
        )

    def observations(self, start_date: str | None = None) -> pd.DataFrame:
        sql = """
            SELECT source, dataset, series_id, series_name, date, value, unit
            FROM observations
        """
        params = []
        if start_date:
            sql += " WHERE date >= ?"
            params.append(start_date)
        return self.conn.execute(sql, params).df()

    def catalog(self) -> pd.DataFrame:
        return self.conn.execute("""
            SELECT source, dataset, series_id, ANY_VALUE(series_name) AS series_name,
                   MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS observations
            FROM observations
            GROUP BY source, dataset, series_id
            ORDER BY source, dataset, series_name
        """).df()

    def auctions(self, start_date: str | None = None) -> pd.DataFrame:
        sql = "SELECT * EXCLUDE (ingested_at, raw_json) FROM treasury_auctions"
        params = []
        if start_date:
            sql += " WHERE auction_date >= ?"
            params.append(start_date)
        return self.conn.execute(sql, params).df()
