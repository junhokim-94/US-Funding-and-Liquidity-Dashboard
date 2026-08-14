from __future__ import annotations

from io import StringIO
import json
import logging
import os

import pandas as pd
import requests

from funding_dashboard.http import HttpClient

LOGGER = logging.getLogger(__name__)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"


def _normalize(raw: pd.DataFrame, series_id: str, series_name: str, method: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={"observation_date": "date"})
    if not {"date", "value"}.issubset(raw.columns):
        return pd.DataFrame()
    raw = raw[["date", "value"]].copy()
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["date", "value"])
    raw["source"] = "fred"
    raw["dataset"] = "fred"
    raw["series_id"] = series_id
    raw["series_name"] = series_name
    raw["unit"] = None
    raw["metadata_json"] = json.dumps({"download": method})
    return raw[["source", "dataset", "series_id", "series_name", "date", "value", "unit", "metadata_json"]]


def fetch_series(client: HttpClient, series_id: str, series_name: str, start_date: str) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if api_key:
        response = client.get(
            FRED_API,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
            },
            expire_after=12 * 3600,
        )
        observations = response.json().get("observations", [])
        return _normalize(pd.DataFrame(observations), series_id, series_name, "fred_api")

    response = client.get(
        FRED_CSV,
        params={"id": series_id, "cosd": start_date},
        expire_after=12 * 3600,
    )
    raw = pd.read_csv(StringIO(response.text))
    if len(raw.columns) >= 2:
        raw = raw.iloc[:, :2]
        raw.columns = ["date", "value"]
    return _normalize(raw, series_id, series_name, "fredgraph.csv")


def fetch_many(
    client: HttpClient,
    series: dict[str, str],
    start_date: str | dict[str, str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    frames = []
    failures: list[str] = []
    for series_id, name in series.items():
        series_start = start_date.get(series_id) if isinstance(start_date, dict) else start_date
        try:
            frame = fetch_series(client, series_id, name, series_start)
            if frame.empty:
                failures.append(f"{series_id}: no rows")
            else:
                frames.append(frame)
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as exc:
            failures.append(f"{series_id}: {exc}")
            LOGGER.warning("FRED series %s failed: %s", series_id, exc)
    if failures and strict:
        raise RuntimeError("FRED series failures: " + " | ".join(failures))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
