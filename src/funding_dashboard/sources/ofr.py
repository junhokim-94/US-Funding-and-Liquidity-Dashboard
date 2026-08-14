from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from funding_dashboard.http import HttpClient

LOGGER = logging.getLogger(__name__)
STFM_BASE = "https://data.financialresearch.gov/v1"
HF_BASE = "https://data.financialresearch.gov/hf/v1"


def fetch_mnemonics(client: HttpClient, dataset: str, hedge_fund: bool = False) -> pd.DataFrame:
    base = HF_BASE if hedge_fund else STFM_BASE
    response = client.get(
        f"{base}/metadata/mnemonics",
        params={"dataset": dataset},
        expire_after=7 * 24 * 3600,
    )
    payload = response.json()
    if isinstance(payload, dict):
        payload = payload.get(dataset, [])
    return pd.DataFrame(payload)


def _extract_points(node: dict[str, Any]) -> list:
    ts = node.get("timeseries", {})
    if isinstance(ts, dict):
        for key in ("aggregation", "data", "observations"):
            if isinstance(ts.get(key), list):
                return ts[key]
    for key in ("aggregation", "data", "observations"):
        if isinstance(node.get(key), list):
            return node[key]
    return []


def normalize_dataset(payload: dict[str, Any], dataset: str, source: str = "ofr") -> pd.DataFrame:
    rows: list[dict] = []
    series_root = payload.get("timeseries", payload.get("series", {}))
    if not isinstance(series_root, dict):
        return pd.DataFrame()
    for mnemonic, node in series_root.items():
        if not isinstance(node, dict):
            continue
        nested_meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        series_name = (
            node.get("series_name")
            or node.get("name")
            or node.get("long_name")
            or node.get("title")
            or nested_meta.get("series_name")
            or nested_meta.get("name")
            or nested_meta.get("long_name")
            or nested_meta.get("title")
            or mnemonic
        )
        metadata = {k: v for k, v in node.items() if k not in {"timeseries", "aggregation", "data", "observations"}}
        if nested_meta:
            metadata.update(nested_meta)
        for point in _extract_points(node):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                d, value = point[0], point[1]
            elif isinstance(point, dict):
                d = point.get("date") or point.get("time")
                value = point.get("value")
            else:
                continue
            if value is None:
                continue
            rows.append({
                "source": source,
                "dataset": dataset,
                "series_id": mnemonic,
                "series_name": series_name,
                "date": d,
                "value": value,
                "unit": metadata.get("unit") or metadata.get("units"),
                "metadata_json": json.dumps(metadata, default=str),
            })
    return pd.DataFrame(rows)


def fetch_dataset(client: HttpClient, dataset: str, start_date: str, hedge_fund: bool = False) -> pd.DataFrame:
    base = HF_BASE if hedge_fund else STFM_BASE
    response = client.get(
        f"{base}/series/dataset",
        params={"dataset": dataset, "start_date": start_date, "remove_nulls": "true"},
        expire_after=12 * 3600,
    )
    return normalize_dataset(
        response.json(),
        dataset=dataset,
        source="ofr_hf" if hedge_fund else "ofr",
    )
